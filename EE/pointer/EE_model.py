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
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    BertModel,
    BertForSequenceClassification
)
import numpy as np
import re
import json
from evaluation import evaluate,evaluate_bool
from evaluation import F1Counter

class RoPE(object):
    def __init__(self):
        self.cache={}

    def pos_embedding(self,iodim):
        if iodim in self.cache:
            return self.cache[iodim]
        seq_len,out_dim = iodim
        # [] -> [[]]   
        position_ids = torch.arange(0,seq_len,dtype=torch.float)[None]

        indices = torch.arange(0, out_dim//2, dtype=torch.float)
        indices = torch.pow(10000.0, -2*indices/out_dim)
        embeddings = torch.einsum("bn,d->bnd", position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)],dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, out_dim))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        embeddings = embeddings.to(device)
        self.cache[iodim]=embeddings
        return embeddings
    
    def add_pos_embedding(self,qw):
        pos=self.pos_embedding((qw.shape[1],qw.shape[-1]))
        cos_pos = torch.repeat_interleave(pos[..., 1::2],2,dim=-1)
        sin_pos = torch.repeat_interleave(pos[..., ::2],2,dim=-1)
        qw2=torch.stack([-qw[...,1::2], qw[...,::2]],dim=-1)
        qw2=torch.reshape(qw2, qw.shape)
        qw=qw*cos_pos+qw2*sin_pos
        return qw


class PointerNet(torch.nn.Module):
    def __init__(self,config,label_categories_num):
        super(PointerNet, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        self.cls_dense=torch.nn.Linear(self.encoder.config.hidden_size,1)

        # Pointer Network: one label(event role) has both start and end pointer
        self.pointer_danse=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)

        self.dropout=torch.nn.Dropout(config.dropout)

        # self.sec_danse1=torch.nn.Linear(self.encoder.config.hidden_size,256)
        # self.sec_danse2=torch.nn.Linear(self.encoder.config.hidden_size,256)
        # Binary classification: is the pointer?
        self.activation=torch.sigmoid
        # self.rope=RoPE()

    def forward(self,x):
        embedding=self.encoder(**x)[0]
        embedding=self.dropout(embedding)

        event_detector=self.activation(self.cls_dense(embedding[:,0]))
        event_detector=event_detector.reshape(event_detector.shape[0],1,event_detector.shape[-1])

        pointer=self.activation(self.pointer_danse(embedding))
        pointer=pointer*event_detector

        # # SEC:Start End Concurrence 
        # start_embedding=self.sec_danse1(embedding)
        # start_embedding=self.rope.add_pos_embedding(start_embedding)
        # start_embedding=self.dropout(start_embedding)

        # end_embedding=self.sec_danse2(embedding)
        # end_embedding=self.rope.add_pos_embedding(end_embedding)
        # end_embedding=self.dropout(end_embedding)

        # sec=self.activation(torch.matmul(start_embedding,end_embedding.transpose(1,2)))
        # sec=sec*event_detector
        return pointer

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

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        self.processor = NERProcessor(config, self.tokenizer)

        self.labels = len(self.processor.event_schema.id2labels)

        self.model = PointerNet(config,self.labels)
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

        self.train_loader = self.processor.create_dataloader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = self.processor.create_dataloader(
            dev_data, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,sec_tensor,label_text_index = batch
        pointer = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0], label_tensors[attention_mask!=0])
        # sec_loss= self.criterion(torch.triu(sec,1),sec_tensor)
        loss=pointer_loss

        self.log('pointer_loss', pointer_loss.item(),prog_bar=True)
        # self.log('sec_loss', sec_loss.item(),prog_bar=True)

        self.log('train_loss', loss.item())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,sec_tensor,label_text_index = batch
        pointer = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
        # sec_loss= self.criterion(torch.triu(sec,1),sec_tensor)
        loss=pointer_loss

        pointer_counter=F1Counter()
        evaluate_bool(torch.where(pointer[attention_mask!=0]>0.5,1,0),label_tensors[attention_mask!=0],pointer_counter)

        # sec_counter=F1Counter()
        # evaluate_bool(torch.where(torch.triu(sec,1)>0.5,1,0),sec_tensor,sec_counter)

        span_pred=self.processor.from_label_tensor_to_label_index(pointer,offset_mapping)
        span_counter=F1Counter()
        evaluate(span_pred,label_text_index,span_counter)

        return pointer_loss.item(),pointer_counter,span_counter


    def validation_epoch_end(self, outputs):
        span_total_counter=F1Counter()
        pointer_total_counter=F1Counter()
        # sec_total_counter=F1Counter()

        pointer_total_losss=0
        # sec_total_loss=0
        for output in outputs:
            pointer_loss,pointer_counter,span_counter= output

            pointer_total_counter+=pointer_counter
            # sec_total_counter+=sec_counter
            span_total_counter+=span_counter

            pointer_total_losss+=pointer_loss
            # sec_total_loss+=sec_loss

        precision,recall,f1=pointer_total_counter.cal_score()
        print(f"Finished epoch  pointer loss:{pointer_total_losss}, pointer precisoin:{precision}, pointer recall:{recall},pointer f1:{f1}")

        # precision,recall,f1=sec_total_counter.cal_score()
        # print(f"Finished epoch  sec loss:{sec_total_loss}, sec precisoin:{precision}, sec recall:{recall},sec f1:{f1}")

        precision,recall,f1=span_total_counter.cal_score()
        print(f"Finished epoch  span precisoin:{precision}, span recall:{recall},span f1:{f1}")


        self.log('val_loss', pointer_total_losss)
        self.log('val_pre', torch.tensor(precision))
        self.log('val_rec', torch.tensor(recall))
        self.log('val_f1', torch.tensor(f1))


class NERPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert
        self.use_crf = config.use_crf
        
        print('load checkpoint:', checkpoint_path)
        self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)

    def extract_events(self,text,offset_mapping,preds):
        events_dict={}
        for index,label_id in preds:
            argument=text[index[0]:index[1]]
            label=self.model.processor.event_schema.id2labels[label_id[0]]
            label=re.match("B-(.+):(.+)",label).groups()
            events_dict.setdefault(label[0],[])
            if argument:
                events_dict[label[0]].append((label[1],argument))

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
        self.test_data = self.model.processor.get_test_data()
        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
        self.test_data, batch_size=2, shuffle=False)

        print("The TEST num is:", len(self.test_data))

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0

        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)-1):
                    batch[i] = batch[i].to(device)

                input_ids, token_type_ids, attention_mask,offset_mapping,label_tensors,sec_tensor,label_index = batch

                pointer = self.model(input_ids, token_type_ids, attention_mask)

                preds=self.model.processor.from_label_tensor_to_label_index(pointer,offset_mapping)

                for offset,pred in zip(offset_mapping,preds):
                    item=dict(self.test_data[cnt].items())
                    events=self.extract_events(item["text"],offset,pred)
                    item["event_list"]=events
            
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")
                    cnt+=1

        print('done--all %d tokens.' % cnt)
    
    def validation(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0
        span_total_counter=F1Counter()
        pointer_total_counter=F1Counter()
        sec_total_counter=F1Counter()

        dev_data = self.model.processor.get_dev_data()
        dataloader = self.model.processor.create_dataloader(dev_data, 8, shuffle=False)
        for batch in tqdm.tqdm(dataloader):
            for i in range(len(batch)-1):
                batch[i] = batch[i].to(device)

            input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,sec_tensor,label_text_index = batch

            pointer,sec = self.model(input_ids, token_type_ids, attention_mask)

            pointer_counter=F1Counter()
            evaluate_bool(torch.where(pointer[attention_mask!=0]>0.5,1,0),label_tensors[attention_mask!=0],pointer_counter)

            sec_counter=F1Counter()
            evaluate_bool(torch.where(torch.triu(sec,1)>0.5,1,0),sec_tensor,sec_counter)

            span_pred=self.model.processor.from_label_tensor_to_label_index(pointer,sec,offset_mapping)
            span_counter=F1Counter()
            evaluate(span_pred,label_text_index,span_counter)

            pointer_total_counter+=pointer_counter
            sec_total_counter+=sec_counter
            span_total_counter+=span_counter
        
        precision,recall,f1=pointer_total_counter.cal_score()
        print(f"Finished epoch pointer precisoin:{precision}, pointer recall:{recall},pointer f1:{f1}")

        precision,recall,f1=sec_total_counter.cal_score()
        print(f"Finished epoch sec precisoin:{precision}, sec recall:{recall},sec f1:{f1}")

        precision,recall,f1=span_total_counter.cal_score()
        print(f"Finished epoch  span precisoin:{precision}, span recall:{recall},span f1:{f1}")

