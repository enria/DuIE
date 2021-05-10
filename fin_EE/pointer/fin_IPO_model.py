# coding=utf-8
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
from evaluation import evaluate_acc
from fin_IPO_data_util import IPOProcessor

class IPOModel(torch.nn.Module):
    def __init__(self,config,IPO_categories_num):
        super(IPOModel, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        self.cls_dense=torch.nn.Linear(self.encoder.config.hidden_size,IPO_categories_num)
        
        self.activation=torch.softmax
        self.dropout=torch.nn.Dropout(config.dropout)

    def forward(self,x):
        embedding =self.encoder(**x)[0]
        cls_embedding=embedding[:,0]
        dropout_cls=self.dropout(cls_embedding)

        IPO_cls=self.cls_dense(dropout_cls)

        return IPO_cls

class IPOModule(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(IPOModule, self).__init__()

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        self.processor = IPOProcessor(config, self.tokenizer)

        self.labels = len([
                "筹备上市",
                "暂停上市",
                "正式上市",
                "终止上市"
            ])
        self.model = IPOModel(config,self.labels)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        print("Pointer Network model init: done.")
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        IPO_cls=self.model({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask})
        return IPO_cls
    
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


        # import random
        # random.shuffle(train_data)
        # random.shuffle(dev_data)
        # train_data=train_data[:100]
        # dev_data=train_data[:100]

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
        input_ids, token_type_ids, attention_mask, label_tensors= batch
        IPO_cls= self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        cls_loss = self.criterion(IPO_cls, label_tensors)
        loss=cls_loss

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_tensors= batch
        IPO_cls= self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        cls_loss = self.criterion(IPO_cls, label_tensors)
        loss=cls_loss

        pred_cls=torch.argmax(IPO_cls,dim=-1)

        cls_acc=evaluate_acc(pred_cls,label_tensors)

        return cls_loss,cls_acc


    def validation_epoch_end(self, outputs):
        cls_totol_loss,cls_total_acc=0,(0,0)
        for output in outputs:
            cls_loss,cls_acc = output
            cls_totol_loss+=cls_loss.item()
            cls_total_acc=(cls_total_acc[0]+cls_acc[0],cls_total_acc[1]+cls_acc[1])

        acc=cls_total_acc[0]/cls_total_acc[1]
        self.log('val_acc', acc)

        print(f"\nFinished epoch cls loss:{cls_totol_loss:.4f}, acc:{acc:.4f}")

class IPOPredictor:
    def __init__(self, checkpoint_path, config):

        self.model = IPOModule.load_from_checkpoint(checkpoint_path, config=config)

        self.test_data = self.model.processor.get_test_data(config.test_path)

        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
            self.test_data, batch_size=config.batch_size, shuffle=False)
        
        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

        self.config=config


    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt=0

        item_IPO_dict={}
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)-1):
                batch[i] = batch[i].to(device)

            input_ids, token_type_ids, attention_mask, label_tensors = batch

            IPO_cls = self.model(input_ids, token_type_ids, attention_mask)
            preds=torch.argmax(IPO_cls,dim=-1)

            for pred in preds:
                item=dict(self.test_data[cnt].items())
                item_IPO_dict[item["id"]]=self.model.processor.IPO_categories[pred]
                cnt+=1
                
        with open(outfile_txt, 'w') as fout,open(self.config.test_path) as test_data:
            for line in test_data:
                item=json.loads(line)
                if item["id"] in item_IPO_dict:
                    for event in item["event_list"]:
                        if event["event_type"]=="公司上市":
                            for role in event["arguments"]:
                                if role["role"]=="环节":
                                    break
                            else:
                                event["arguments"].append({"role":"环节","argument":item_IPO_dict[item["id"]]})
                
                fout.write(json.dumps(item,ensure_ascii=False)+"\n")                

        print('done--all %d tokens.' % cnt)