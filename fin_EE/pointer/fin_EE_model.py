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
from evaluation import evaluate_pointer,evaluate_concurrence
from evaluation import F1Counter
from fin_EE_data_util import NERProcessor,EventSchemaDict


class PointerNet(torch.nn.Module):
    def __init__(self,config,label_categories_num):
        super(PointerNet, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        # Pointer Network: one label(event role) has both start and end pointer
        self.dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)
        
        # 共现矩阵 (w,h)*(h,h)->(w,h), (w,h)*(h,w)->(w,w)
        self.dense2=torch.nn.Linear(self.encoder.config.hidden_size,self.encoder.config.hidden_size)

        # Binary classification: is the pointer?
        self.activation=torch.sigmoid

    def forward(self,x):
        embedding =self.encoder(**x)[0]
        pointer=self.activation(self.dense(embedding))

        fixed_embedding=embedding.detach()
        concurrence=self.activation(torch.matmul(embedding,self.dense2(fixed_embedding).transpose(1,2)))
        # concurrence=self.activation(torch.matmul(embedding,embedding.transpose(1,2)))
        return pointer,concurrence

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
        pointer,concurrence=self.model({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask})
        return pointer,concurrence
    
    def configure_optimizers(self):
        arg_list = [p for p in self.parameters() if p.requires_grad]
        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def prepare_data(self):
        origin_train_data = self.processor.get_train_data()
        origin_dev_data=self.processor.get_dev_data()
        train_data=self.processor.process_long_text(origin_train_data,512)
        dev_data=self.processor.process_long_text(origin_dev_data,512)

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
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
        # concurrence_loss= self.criterion(concurrence[attention_mask!=0,:], concurrence_tensors[attention_mask!=0])
        concurrence_loss=torch.tensor(0)
        loss=pointer_loss+concurrence_loss

        self.log('pointer_loss', pointer_loss.item(),prog_bar=True)
        self.log('concurrence_loss', concurrence_loss.item(),prog_bar=True)
        self.log('train_loss', loss.item())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
        # concurrence_loss= self.criterion(concurrence[attention_mask!=0,:], concurrence_tensors[attention_mask!=0])
        concurrence_loss=torch.tensor(0)
        loss=pointer_loss+concurrence_loss

        pointer_pred=self.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)
        pointer_pred,_=zip(*pointer_pred)
        concurrence_pred=torch.where(concurrence > 0.5, 1, 0)

        pointer_counter=F1Counter()
        evaluate_pointer(pointer_pred,label_text_index,pointer_counter)
        concurrence_counter=F1Counter()
        evaluate_concurrence(concurrence_pred,concurrence_tensors,concurrence_counter)

        return pointer_loss,concurrence_loss,pointer_counter,concurrence_counter


    def validation_epoch_end(self, outputs):
        pointer_total_counter=F1Counter()
        concurrence_total_counter=F1Counter()
        pointer_totol_losss,concurrence_total_loss=0,0
        for output in outputs:
            pointer_loss,concurrence_loss,pointer_counter,concurrence_counter = output
            pointer_totol_losss+=pointer_loss.item()
            concurrence_total_loss+=concurrence_loss.item()
            pointer_total_counter+=pointer_counter
            concurrence_total_counter+=concurrence_counter

        precision,recall,f1=pointer_total_counter.cal_score()
        print(f"Finished epoch pointer_loss:{pointer_totol_losss}, pointer precisoin:{precision}, pointer recall:{recall},pointer f1:{f1}")
        precision,recall,f1=concurrence_total_counter.cal_score()
        print(f"Finished epoch concurrence_loss:{concurrence_total_loss}, concurrence precisoin:{precision}, concurrence recall:{recall},concurrence f1:{f1}")

        precision,recall,f1=pointer_total_counter.cal_score()
        self.log('val_pre', torch.tensor(precision))
        self.log('val_rec', torch.tensor(recall))
        self.log('val_f1', torch.tensor(f1))


class NERPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert
        self.use_crf = config.use_crf

        self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)

        self.origin_test_data = self.model.processor.get_test_data()
        self.test_data=self.model.processor.process_long_text(self.origin_test_data,512)

        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
            self.test_data, batch_size=config.batch_size, shuffle=False)
        
        # 保存每一条数据的事件
        self.item_events={}
        self.event_schema=EventSchemaDict(config.schema_path)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)


    def extract_events(self,item,offset_mapping,preds):
        item_id=item["id"]
        self.item_events.setdefault(item_id,{})
        events_dict=self.item_events[item_id]

        for index,label_id in preds:
            argument=item["text"][index[0]:index[1]]
            label=self.model.processor.event_schema.id2labels[label_id[0]]
            event_type,role=re.match("B-(.+):(.+)",label).groups()
            if role=="TRG":
                continue
            events_dict.setdefault(event_type,set())
            if argument:
                if item.get("slice",0)==0:
                    events_dict[event_type].add((role,argument,index))
                else:
                    if index[0]<item["title_length"]:
                        events_dict[event_type].add((role,argument,index))
                    else:
                        shift_index=(index[0]+item["text_begin_index"],
                                     index[1]+item["text_begin_index"])
                        events_dict[event_type].add((role,argument,shift_index))

    def find_closest(self,subject,roles,threshold=200):
        closest_index=-1
        subject_start=subject[2][0]
        for index,role in enumerate(roles):
            if abs(role[2][0]-subject_start)>threshold:
                continue
            if closest_index==-1 or \
                abs(roles[closest_index][2][0]-subject_start)>abs(role[2][0]-subject_start):
                closest_index=index
        if closest_index > -1:
            return roles[closest_index]
        return None

    def cluster_events(self,events_dict,concurrence):
        new_events=[]
        for event_type,roles in events_dict.items():
            subjects=[]
            assist_roles={}

            # for role in roles:
            #     if role[0] in self.model.processor.event_schema.event_subject[event_type]:
            #         subjects.append(role)
            #     else:
            #         assist_roles.setdefault(role[0],[])
            #         assist_roles[role[0]].append(role)
                    
            # event_role_cluster=[]
            # if len(subjects)>1:
            #     for subject in subjects:
            #         clusetr=[subject]
            #         for role_type,roles in assist_roles.items():
            #             for role in roles:
            #                 if role[-1][0] in concurrence.get(subject[-1][0],set()):
            #                     clusetr.append(role)
            #         event_role_cluster.append(clusetr)
            # else:
            #     event_role_cluster.append(roles)
            
            # TODO 
            event_role_cluster=[roles]
            
            for role_cluster in event_role_cluster:
                new_events.append({
                                    "event_type":event_type,
                                    "arguments":[{
                                        "role":role[0],
                                        "argument":role[1]
                                    } for role in role_cluster]
                                  })
        return new_events

    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0
        concurrence_dict={}
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)-1):
                batch[i] = batch[i].to(device)

            input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch

            pointer,concurrence = self.model(input_ids, token_type_ids, attention_mask)
            preds=self.model.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)

            for offset,pred_role in zip(offset_mapping,preds):
                item=dict(self.test_data[cnt].items())
                self.extract_events(item,offset,pred_role[0])
                cnt+=1
                concurrence_dict[item["id"]]=pred_role[1]
                
        with open(outfile_txt, 'w') as fout:
            for item in tqdm.tqdm(self.origin_test_data):
                    events_dict=self.item_events.get(item["id"],{})
                    events=self.cluster_events(events_dict,concurrence_dict[item["id"]])
                    item=dict(item.items())
                    item["event_list"]=events
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")

        print('done--all %d tokens.' % cnt)
