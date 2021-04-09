# coding=utf-8
from EE_data_util import NERProcessor
import pytorch_lightning as pl
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import csv
from conlleval import evaluate
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    BertModel
)
import numpy as np
import re
import json
# from model_layers import LSTM_attention, multihead_attention

# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NERModel(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(NERModel, self).__init__()
        

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.crf_lr = config.crf_lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.use_bert = config.use_bert
        self.use_crf = config.use_crf
        
        self.layerNorm = torch.nn.LayerNorm
        # 2. Init Model
        if self.use_bert: # BERT模型
            self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
            special_tokens_dict = { 'additional_special_tokens': ["<COM>", "“", "”"] }  # 在词典中增加特殊字符
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.processor = NERProcessor(config, self.tokenizer)

            self.labels = len(self.processor.event_schema.id2labels)

            self.model = BertForTokenClassification.from_pretrained(
                config.pretrained_path, num_labels=self.labels)
            self.model.resize_token_embeddings(len(self.tokenizer))
            print('num tokens:', len(self.tokenizer))

        # 2. Init crf model and loss
        if self.use_crf:
            # crf doc: https://pytorch-crf.readthedocs.io/en/stable/
            self.crf = CRF(num_tags=self.labels, batch_first=True)
        else:
            # weight = torch.Tensor([1, 100, 50])
            # self.criterion = nn.CrossEntropyLoss(weight=weight)
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.NLLLoss(ignore_index=0, size_average=True)

        print("NER model init: done.")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.use_bert:
            feats = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )[0]  # [bs,len,numlabels]

        return feats

    def training_step(self, batch, batch_idx):
        if self.use_bert:
            input_ids, token_type_ids, attention_mask,offset_mapping, labels = batch
            feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.use_crf:
            loss = -self.crf(feats, labels,
                             mask=attention_mask.byte(), reduction='mean')
            # pred = self.crf.decode(feats)
            # acc = (torch.tensor(pred).cuda() == labels).float().mean()
        else:
            loss = self.criterion(feats[attention_mask!=0,:], labels[attention_mask!=0])
            # pred = feats.argmax(dim=-1)
            # acc = (pred == labels).float().mean()

        # tensorboard_logs = {'train_loss': loss}
        # return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.use_bert:
            input_ids, token_type_ids, attention_mask,offset_mapping, labels = batch
            feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        if self.use_crf:
            loss = -self.crf(feats, labels,
                             mask=attention_mask.byte(), reduction='mean')
            pred = self.crf.decode(feats)
            # acc = (torch.tensor(pred).cuda() == labels).float().mean()
        else:
            loss = self.criterion(feats[attention_mask!=0,:], labels[attention_mask!=0])
            pred = feats.argmax(dim=-1)
            # acc = (pred == labels).float().mean()

        gold = labels[attention_mask != 0]
        pre = torch.tensor(pred).cuda()[attention_mask != 0]

        # return {'val_loss': loss, 'gold': gold, 'pre': pre}
        return loss,gold,pre

    def prepare_data(self):
        train_data = self.processor.get_train_data()
        dev_data=self.processor.get_dev_data()

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        if self.use_bert:
            self.train_loader = self.processor.create_dataloader(
                train_data, batch_size=self.batch_size, shuffle=True)
            self.valid_loader = self.processor.create_dataloader(
                dev_data, batch_size=self.batch_size, shuffle=False)
            # self.test_loader = self.processor.create_dataloader(val_data, batch_size=self.batch_size, shuffle=False)

    def validation_epoch_end(self, outputs):

        val_loss,gold,pre=zip(*outputs)
        
        val_loss = torch.stack(val_loss).mean().cpu()
        gold = torch.cat(gold).cpu()
        pre = torch.cat(pre).cpu()

        true_seqs = [self.processor.event_schema.id2labels[int(g)] for g in gold]
        pred_seqs = [self.processor.event_schema.id2labels[int(g)] for g in pre]

        for i in range(len(true_seqs)):
            if true_seqs[i].endswith(":TRG"):
                true_seqs[i]="O"
            if pred_seqs[i].endswith(":TRG"):
                pred_seqs[i]="O"

        print('\n')
        prec, rec, f1 = evaluate(true_seqs, pred_seqs)
        # tensorboard_logs = {'val_loss': val_loss, 'val_pre': torch.tensor(prec),
        #                     'val_rec': rec, 'val_f1': torch.tensor(f1)}
        # return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

        self.log('val_loss', val_loss)
        self.log('val_pre', torch.tensor(prec))
        self.log('val_rec', rec)
        self.log('val_f1', torch.tensor(f1))

    def configure_optimizers(self):
        if self.use_crf:
            crf_params_ids = list(map(id, self.crf.parameters()))
            base_params = filter(lambda p: id(p) not in crf_params_ids, [
                                 p for p in self.parameters() if p.requires_grad])

            arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]
        else:
            # label_embed_and_attention_params = list(map(id, self.label_embedding.parameters())) + list(map(id, self.self_attention.parameters()))
            # arg_list = [{'params': list(self.label_embedding.parameters()) + list(self.self_attention.parameters()), 'lr': self.lr}]
            arg_list = [p for p in self.parameters() if p.requires_grad]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class NERPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert
        self.use_crf = config.use_crf

        self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)

        self.test_data = self.model.processor.get_test_data()
        if config.use_bert:
            self.tokenizer = self.model.tokenizer
            self.dataloader = self.model.processor.create_dataloader(
                self.test_data, batch_size=config.batch_size, shuffle=False)


        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def extract_events(self,text,offset_mapping,preds):
        events_dict={}
        index=1 #直接跳过CLS
        last_B,last_B_index=0,0
        while index<len(preds):
            if last_B:
                if preds[index]!=last_B+1:
                    argument=text[offset_mapping[last_B_index][0]:offset_mapping[index-1][1]]
                    label=self.model.processor.event_schema.id2labels[last_B]
                    label=re.match("B-(.+):(.+)",label).groups()
                    events_dict.setdefault(label[0],[])
                    if argument:
                        events_dict[label[0]].append((label[1],argument))
                    last_B,last_B_index=0,0

            if preds[index]!=0 and preds[index]%2:
                last_B,last_B_index=preds[index],index
            index+=1
        events=[]
        for event_type,roles in events_dict.items():
            if len(roles)==1 and roles[0][0].endswith("TRG"):
                continue
            events.append({
                "event_type":event_type,
                "arguments":[{
                    "role":role[0],
                    "argument":role[1]
                } for role in roles if not role[0].endswith("TRG")]
            })
        return events


    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0

        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)
                if self.use_bert:
                    input_ids, token_type_ids, attention_mask,offset_mapping,labels = batch

                emissions = self.model(input_ids, token_type_ids, attention_mask)

                if self.use_crf:
                    preds = self.model.crf.decode(emissions)  # list
                else:
                    preds = emissions.argmax(dim=-1).cpu()


                for offset,pred in zip(offset_mapping,preds):
                    item=dict(self.test_data[cnt].items())
                    events=self.extract_events(item["text"],offset,pred)
                    item["event_list"]=events
            
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")
                    cnt+=1

        print('done--all %d tokens.' % cnt)
