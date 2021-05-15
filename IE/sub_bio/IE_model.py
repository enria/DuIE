# coding=utf-8
from random import sample
import pytorch_lightning as pl
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    BertModel
)
import numpy as np
import re
import json
from evaluation import evaluate_pointer,evaluate_concurrence,evaluate_span
from evaluation import F1Counter
from IE_data_util import SubjectProcessor,SubjectSchemaDict


class SRecModel(torch.nn.Module):
    def __init__(self,config,label_categories_num):
        super(SRecModel, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        # Pointer Network: one label(event role) has both start and end pointer
        self.dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)
        
        # Binary classification: is the pointer?
        self.activation=torch.sigmoid
        self.dropout=torch.nn.Dropout(config.dropout)

    def forward(self,x):
        embedding =self.encoder(**x)[0]
        dropout_embedding=self.dropout(embedding)
        pointer=self.activation(self.dense(dropout_embedding))

        return pointer

class SubjectRecModule(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(SubjectRecModule, self).__init__()
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.use_bert = config.use_bert
        
        self.layerNorm = torch.nn.LayerNorm

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        self.processor = SubjectProcessor(config, self.tokenizer)

        self.labels = len(self.processor.subject_schema.id2labels)

        self.model = SRecModel(config,self.labels)
        self.criterion = nn.BCELoss(reduction="sum")

        print("Pointer Network model init: done.")
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pointer=self.model({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask})
        return pointer
    
    def configure_optimizers(self):
        arg_list = [p for p in self.parameters() if p.requires_grad]
        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def prepare_data(self):
        train_data = self.processor.get_train_data()
        dev_data=self.processor.get_dev_data()

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_set = self.processor.create_dataset(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_set = self.processor.create_dataset(
            dev_data, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        if self.config.sample:
            random_sampler=torch.utils.data.RandomSampler(self.train_set,replacement=True,num_samples=20000)
            return torch.utils.data.DataLoader(
                self.train_set,
                shuffle=False,
                batch_size=self.config.batch_size,
                num_workers=4,
                collate_fn=self.train_set.collate_fn,
                sampler=random_sampler
            )
        else:
            return torch.utils.data.DataLoader(
                self.train_set,
                shuffle=False,
                batch_size=self.config.batch_size,
                num_workers=4,
                collate_fn=self.train_set.collate_fn
            )

    def val_dataloader(self):
        if self.config.sample:
            random_sampler=torch.utils.data.RandomSampler(self.valid_set,replacement=True,num_samples=4000)
            return torch.utils.data.DataLoader(
                self.valid_set,
                shuffle=False,
                batch_size=self.config.batch_size,
                num_workers=4,
                collate_fn=self.valid_set.collate_fn,
                sampler=random_sampler
            )
        else:
            return torch.utils.data.DataLoader(
                self.valid_set,
                shuffle=False,
                batch_size=self.config.batch_size,
                num_workers=4,
                collate_fn=self.valid_set.collate_fn
            ) 

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,label_text_index = batch
        pointer = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0], label_tensors[attention_mask!=0])
        loss=pointer_loss

        self.log('pointer_loss', pointer_loss.item(),prog_bar=True)
        self.log('train_loss', loss.item())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,label_text_index = batch
        pointer = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0], label_tensors[attention_mask!=0])
        loss=pointer_loss

        span_pred=self.processor.from_label_tensor_to_label_index(pointer,offset_mapping)
        pointer_counter=F1Counter()
        evaluate_pointer(torch.where(pointer[attention_mask!=0]>0.5,1,0),label_tensors[attention_mask!=0],pointer_counter)

        span_counter=F1Counter()
        evaluate_span(span_pred,label_text_index,span_counter)

        return pointer_loss,pointer_counter,span_counter


    def validation_epoch_end(self, outputs):
        pointer_total_counter=F1Counter()
        span_total_counter=F1Counter()
        pointer_totol_loss=0
        for output in outputs:
            pointer_loss,pointer_counter,span_counter = output
            pointer_totol_loss+=pointer_loss.item()
            pointer_total_counter+=pointer_counter
            span_total_counter+=span_counter

        precision,recall,pf1=pointer_total_counter.cal_score()
        print(f"Finished epoch pointer_loss:{pointer_totol_loss:.4f}, pointer precisoin:{precision:.4f}, pointer recall:{recall:.4f},pointer f1:{pf1:.4f}")
        precision,recall,sf1=span_total_counter.cal_score()
        print(f"Finished epoch span precisoin:{precision:.4f}, span recall:{recall:.4f},span f1:{sf1:.4f}")

        self.log('val_f1', torch.tensor(pf1))

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
          return list(obj)
        return json.JSONEncoder.default(self, obj)

class SubjectPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert

        self.model = SubjectRecModule.load_from_checkpoint(checkpoint_path, config=config)

        test_data = self.model.processor.get_dev_data()
        # dev_data=self.model.processor.get_dev_data()[:500]

        self.test_data = test_data
        self.tokenizer = self.model.tokenizer
        self.dataset= self.model.processor.create_dataset(self.test_data, batch_size=config.batch_size, shuffle=False,is_test=True)

        self.dataloader=torch.utils.data.DataLoader(
                self.dataset,
                shuffle=False,
                batch_size=config.batch_size,
                num_workers=4,
                collate_fn=self.dataset.collate_fn
            ) 
        
        self.subject_schema=self.model.processor.subject_schema

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def extract_subjects(self,item,preds):
        subjects=[]
        for index,label_id in preds:
            argument=item["text"][index[0]:index[1]]
            if not argument:
                continue
            label=self.subject_schema.id2labels[label_id[0]]
            spo_type,role=re.match("B-(.+):(.+)",label).groups()
            if argument:
                subjects.append({
                    "subject":argument,
                    "predicate":spo_type
                })
        return subjects

    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        pointer_total_counter=F1Counter()
        span_total_counter=F1Counter()

        cnt = 0
        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)-1):
                    batch[i] = batch[i].to(device)

                input_ids, token_type_ids, attention_mask,offset_mapping,label_tensors,label_text_index = batch

                pointer = self.model(input_ids, token_type_ids, attention_mask)
                preds=self.model.processor.from_label_tensor_to_label_index(pointer,offset_mapping)

                # pointer_counter=F1Counter()
                # evaluate_pointer(torch.where(pointer[attention_mask!=0]>0.5,1,0),label_tensors[attention_mask!=0],pointer_counter)

                # span_counter=F1Counter()
                # evaluate_span(preds,label_text_index,span_counter)

                # pointer_total_counter+=pointer_counter
                # span_total_counter+=span_counter

                for pred_role in preds:
                    item=dict(self.test_data[cnt].items())
                    subjects=self.extract_subjects(item,pred_role)
                    item["spo_list"]=subjects
                    fout.write(json.dumps(item,ensure_ascii=False,cls=SetEncoder)+"\n")
                    cnt+=1
        # precision,recall,pf1=pointer_total_counter.cal_score()
        # print(f"Finished epoch pointer precisoin:{precision:.4f}, pointer recall:{recall:.4f},pointer f1:{pf1:.4f}")
        # precision,recall,sf1=span_total_counter.cal_score()
        # print(f"Finished epoch span precisoin:{precision:.4f}, span recall:{recall:.4f},span f1:{sf1:.4f}")
        # print('done--all %d tokens.' % cnt)
    
