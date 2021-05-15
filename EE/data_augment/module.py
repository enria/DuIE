import time,json,os
import torch
import numpy as np
import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig
)
from torch.utils.data import DataLoader
from transformers.optimization import AdamW,get_linear_schedule_with_warmup
from typing import Dict, List, Tuple
from utils import calculate_bleu,lmap

class Seq2SeqDataset(object):
    """A dataset that calls prepare_seq2seq_batch."""

    def __init__(self,tokenizer,data_path):
        super().__init__()
        self.max_source_length = 512
        self.max_target_length = 512
        self.tokenizer = tokenizer

        self.pad_token_id = self.tokenizer.pad_token_id
        self.raw_data_items=[]
        self.load_data_file(data_path)
    
    def load_data_file(self,data_path):
        with open(data_path) as data_file:
            for line in data_file:
                item=json.loads(line)
                context=item["text"]
                for event in item["event_list"]:
                    for argument in event["arguments"]:
                        mention=argument["argument"].strip()
                        labeled_mention=f" <{event['event_type']}-{argument['role']}> {mention} </{event['event_type']}-{argument['role']}>"
                        context=context.replace(mention,labeled_mention)
                self.raw_data_items.append(context)

    def __len__(self):
        return len(self.raw_data_items)

    def __getitem__(self, index) -> Dict[str, str]:

        source_line = self.raw_data_items[index]
        tgt_line = self.raw_data_items[index]

        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index}

    def collate_fn(self, batch):
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer(
            [x["src_texts"] for x in batch],
            add_special_tokens=True,
            max_length=self.max_source_length,
            return_tensors="pt",
            truncation=True,
            padding="longest"
        ).data
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [x["tgt_texts"] for x in batch],
                add_special_tokens=True,
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt",
                padding="longest"
            ).data
        batch_encoding["labels"] = labels["input_ids"]
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])

        return batch_encoding

class ARModule(pl.LightningModule):
    """ AutoRegression Module
        aiming to generate input 
    """
    loss_names = ["loss"]
    metric_names = ["bleu"]
    def __init__(self,config):
        super(ARModule, self).__init__()
        self.config=config
        self.model=AutoModelForSeq2SeqLM.from_pretrained(config.pretrained_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        self.add_label_token()
    
    def add_label_token(self):
        label_tokens = []  # 在词典中增加特殊字符
        with open(self.config.schema_path) as scheme:
            for line in scheme:
                item=json.loads(line)
                for role in item["role_list"]:
                    label_tokens.append(f"<{item['event_type']}-{role['role']}>")
                    label_tokens.append(f"</{item['event_type']}-{role['role']}>")

        self.tokenizer.add_tokens(label_tokens)  # 在词典中增加特殊字符
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.vocab_size = len(self.tokenizer)
    
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        decoder_input_ids = self.model._shift_right(tgt_ids)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        assert lm_logits.shape[-1] == self.vocab_size
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        # logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        # # tokens per batch
        # logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        # logs["bs"] = batch["input_ids"].shape[0]
        # logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        # logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0]}
    
    def _generative_step(self, batch: dict, batch_idx=None, dataloader_idx=None) -> dict:
        t0 = time.time()

        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            # num_beams=self.eval_beams,
            # max_length=self.eval_max_length,
            length_penalty=1.0
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        bleu: Dict = self.calc_generative_metrics(preds, target)
        base_metrics.update(gen_time=gen_time, preds=preds, target=target, bleu=bleu)

        return base_metrics

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)
    
    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_bleu(preds, target)
    
    def validation_epoch_end(self, outputs, prefix="val") -> Dict:

        # val_outputs_folder = "val_outputs"
        # os.system("mkdir -p " + os.path.join(self.hparams.output_dir, val_outputs_folder))

        # output_test_predictions_file = os.path.join(self.hparams.output_dir, val_outputs_folder, "validation_predictions_" +
        #                                             str(self.step_count) + ".txt")
        # output_test_targets_file = os.path.join(self.hparams.output_dir, val_outputs_folder, "validation_targets_" +
        #                                             str(self.step_count) + ".txt")
        # # write predictions and targets for later rouge evaluation.
        # with open(output_test_predictions_file, "w") as p_writer, open(output_test_targets_file, "w") as t_writer:
        #     for output_batch in outputs:
        #         p_writer.writelines(s + "\n" for s in output_batch["preds"])
        #         t_writer.writelines(s + "\n" for s in output_batch["target"])
        #     p_writer.close()
        #     t_writer.close()

        # bleu_info = eval_bleu(self.hparams.data_dir, output_test_predictions_file, 'val')

        # rank_zero_info("%s bleu_info: %s", self.step_count, bleu_info)

        # if bleu_info == -1:
        #     bleu_info = float(bleu_info)
        # else:
        #     bleu_info = float(bleu_info.split(",")[0].split("BLEU = ")[1])

        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.metric_names}
        bleu = generative_metrics["bleu"]

        print(f"Finished Epoch, Total loss: {loss.item()} Blue:{bleu.item()}")
        self.log("loss",loss.item())
        self.log("bleu",bleu.item())
        return 

        # # generative_metrics['bleu'] = bleu_info

        # metric_val = (
        #     generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[
        #         self.val_metric]
        # )
        # metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        # generative_metrics.update({k: v.item() for k, v in losses.items()})
        # losses.update(generative_metrics)
        # all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        # all_metrics["step_count"] = self.step_count
        # self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        # preds = flatten_list([x["preds"] for x in outputs])

        # return {
        #     "bleu": bleu_info,
        #     "log": all_metrics,
        #     "preds": preds,
        #     f"{prefix}_loss": loss,
        #     f"{prefix}_{self.val_metric}": metric_tensor,
        # }
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        effective_batch_size = self.config.batch_size 
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.config.max_epochs
    
    def setup(self, mode):
        self.train_loader = self.get_dataloader(self.config.train_path, self.config.batch_size)


    def get_lr_scheduler(self):
        get_schedule_func = get_linear_schedule_with_warmup
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=500, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]
    
    def get_dataset(self, data_path) -> Seq2SeqDataset:
        dataset = Seq2SeqDataset(
            self.tokenizer,
            data_path
        )
        return dataset

    def get_dataloader(self, data_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(data_path)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=4
        )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(self.config.train_path, batch_size=self.config.batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.config.dev_path, batch_size=self.config.batch_size)