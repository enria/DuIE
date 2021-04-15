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
    BertModel
)
import numpy as np
import re
import json
from evaluation import evaluate
from evaluation import F1Counter


class PointerNet(torch.nn.Module):
    def __init__(self,config,label_categories_num):
        super(PointerNet, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        # Pointer Network: one label(event role) has both start and end pointer
        self.dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)

        # Binary classification: is the pointer?
        self.activation=torch.sigmoid

    def forward(self,x):
        embedding=self.encoder(**x)[0]
        pointer=self.activation(self.dense(embedding))
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
        feats=self.model({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask})
        return feats
    
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
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,label_text_index = batch
        feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        loss = self.criterion(feats[attention_mask!=0,:], label_tensors[attention_mask!=0])

        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, lalabel_tensors,label_text_index = batch
        feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        loss = self.criterion(feats[attention_mask!=0,:], lalabel_tensors[attention_mask!=0])

        pre=self.processor.from_label_tensor_to_label_index(feats,offset_mapping)

        return loss,pre,label_text_index


    def validation_epoch_end(self, outputs):
        counter=F1Counter()
        totol_losss=0
        for output in outputs:
            loss,pre,gold = output
            evaluate(pre,gold,counter)
            totol_losss+=loss.item()

        precision,recall,f1=counter.cal_score()
        print(f"Finished epoch loss:{totol_losss}, precisoin:{precision}, recall:{recall},f1:{f1}")


        self.log('val_loss', totol_losss)
        self.log('val_pre', torch.tensor(precision))
        self.log('val_rec', torch.tensor(recall))
        self.log('val_f1', torch.tensor(f1))


class NERPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert
        self.use_crf = config.use_crf

        self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)

        self.test_data = self.model.processor.get_dev_data()
        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
            self.test_data, batch_size=config.batch_size, shuffle=False)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0

        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)-1):
                    batch[i] = batch[i].to(device)

                input_ids, token_type_ids, attention_mask,offset_mapping,label_tensors,label_index = batch

                feats = self.model(input_ids, token_type_ids, attention_mask)

                preds=self.model.processor.from_label_tensor_to_label_index(feats,offset_mapping)

                for offset,pred in zip(offset_mapping,preds):
                    item=dict(self.test_data[cnt].items())
                    events=self.extract_events(item["text"],offset,pred)
                    item["pred_list"]=events
            
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")
                    cnt+=1

        print('done--all %d tokens.' % cnt)
