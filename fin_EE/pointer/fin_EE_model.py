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
import os
from evaluation import evaluate_pointer,evaluate_concurrence
from evaluation import F1Counter
from fin_EE_data_util import NERProcessor,EventSchemaDict


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

        self.rope=RoPE()

        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        # self.cls_dense=torch.nn.Linear(self.encoder.config.hidden_size,1)

        # Pointer Network: one label(event role) has both start and end pointer
        self.pointer_dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)
        
        # 共现矩阵 (w,h)*(h,h)->(w,h), (w,h)*(h,w)->(w,w)
        self.con_head_dense=torch.nn.Linear(self.encoder.config.hidden_size,256)
        self.con_tail_dense=torch.nn.Linear(self.encoder.config.hidden_size,256)

        # Binary classification: is the pointer?
        self.activation=torch.sigmoid
        self.dropout=torch.nn.Dropout(config.dropout)

    def forward(self,x):
        embedding =self.encoder(**x)[0]
        dropout_embedding=self.dropout(embedding)

        # event_detector=self.activation(self.cls_dense(dropout_embedding[:,0]))
        # event_detector=event_detector.reshape(event_detector.shape[0],1,event_detector.shape[-1])

        # aoembedding = self.rope.add_pos_embedding(embedding)
        
        pointer=self.activation(self.pointer_dense(dropout_embedding))
        # pointer=pointer*event_detector

        start_embedding=embedding
        start_embedding=self.con_head_dense(start_embedding)
        start_embedding=self.dropout(start_embedding)

        end_embedding=embedding
        end_embedding=self.con_tail_dense(end_embedding)
        end_embedding=self.dropout(end_embedding)

        concurrence=self.activation(torch.matmul(start_embedding,end_embedding.transpose(1,2)))
        # concurrence=concurrence*event_detector

        return pointer,concurrence

class NERModel(pl.LightningModule):
    def __init__(self, config,kno=-1):
        # 1. Init parameters
        super(NERModel, self).__init__()
        self.config=config
        self.kno=kno

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
        if self.kno>=0:
            self.prepare_k_data()
            return

        origin_train_data = self.processor.get_train_data()
        origin_dev_data=self.processor.get_dev_data()
        train_data=self.processor.process_long_text(origin_train_data,512)
        dev_data=self.processor.process_long_text(origin_dev_data,512)

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

    def prepare_k_data(self):
        origin_train_data = self.processor.get_train_data()
        origin_dev_data=self.processor.get_dev_data()
        train_data=self.processor.process_long_text(origin_train_data,512)
        dev_data=self.processor.process_long_text(origin_dev_data,512)

        all_data=train_data+dev_data
        all_data_num=len(all_data)
        k_avg_num=all_data_num//self.config.k+1

        k_fold_train_data,k_fold_dev_data=[],None
        for i in range(self.config.k):
            cur_data_split=all_data[i*k_avg_num:(i+1)*k_avg_num]
            if i==self.kno:
                k_fold_dev_data=cur_data_split
            else:
                k_fold_train_data.extend(cur_data_split)
        
        print("train_length:", len(k_fold_train_data))
        print("valid_length:", len(k_fold_dev_data))

        self.train_loader = self.processor.create_dataloader(
            k_fold_train_data, batch_size=self.batch_size, shuffle=False)
        self.valid_loader = self.processor.create_dataloader(
            k_fold_dev_data, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
        concurrence_loss= self.criterion(concurrence[attention_mask!=0,:], concurrence_tensors[attention_mask!=0])
        loss=pointer_loss+concurrence_loss

        self.log('pointer_loss', pointer_loss.item(),prog_bar=True)
        self.log('concurrence_loss', concurrence_loss.item(),prog_bar=True)
        self.log('train_loss', loss.item())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
        concurrence_loss= self.criterion(concurrence[attention_mask!=0,:], concurrence_tensors[attention_mask!=0])
        loss=pointer_loss+concurrence_loss

        pointer_pred=self.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)
        pointer_pred,_=zip(*pointer_pred)
        concurrence_pred=torch.where(concurrence > 0.5, 1, 0)

        pointer_counter=F1Counter()
        evaluate_pointer(pointer_pred,label_text_index,pointer_counter)
        concurrence_counter=F1Counter()
        evaluate_concurrence(torch.where(concurrence[attention_mask!=0]>0.5,1,0),concurrence_tensors[attention_mask!=0],concurrence_counter)

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

        pointer_precision,pointer_recall,pointer_f1=pointer_total_counter.cal_score()
        concurrence_precision,concurrence_recall,concurrence_f1=concurrence_total_counter.cal_score()

        precision,recall,f1=pointer_total_counter.cal_score()
        self.log('pf1', torch.tensor(pointer_f1))
        self.log('cf1', torch.tensor(concurrence_f1))
        self.log('val_total_f1', pointer_f1+concurrence_f1)

        print(f"\nFinished epoch ploss:{pointer_totol_losss:.4f}, pprec:{pointer_precision:.4f}, precall:{pointer_recall:.4f}, pf1:{pointer_f1:.4f}\t"+
              f"closs:{concurrence_total_loss:.4f}, cprec:{concurrence_precision:.4f}, crecall:{concurrence_recall:.4f}, cf1:{concurrence_f1:.4f}")

class NERPredictor:
    def __init__(self, checkpoint_path, config):
        self.use_bert = config.use_bert
        self.use_crf = config.use_crf
        self.config=config

        self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # self.origin_test_data = self.model.processor.get_test_data()
        # self.test_data=self.model.processor.process_long_text(self.origin_test_data,512)

        # self.tokenizer = self.model.tokenizer
        # self.dataloader = self.model.processor.create_dataloader(
        #     self.test_data, batch_size=config.batch_size, shuffle=False)
        
        self.processor=NERProcessor(config)
        
        # 保存每一条数据的事件
        self.item_events={}
        self.event_schema=EventSchemaDict(config.schema_path)

        # print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)


    def extract_events(self,item,offset_mapping,preds):
        item_id=item["id"]
        self.item_events.setdefault(item_id,{})
        events_dict=self.item_events[item_id]

        for index,label_id in preds:
            argument=item["text"][index[0]:index[1]]
            label=self.event_schema.id2labels[label_id[0]]
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

            for role in roles:
                if role[0] in self.event_schema.event_subject[event_type]:
                    subjects.append(role)
                else:
                    assist_roles.setdefault(role[0],[])
                    assist_roles[role[0]].append(role)
                    
            event_role_cluster=[]
            if len(subjects)>1:
                for subject in subjects:
                    clusetr=[subject]
                    for role_type,roles in assist_roles.items():
                        for role in roles:
                            if role[-1][0] in concurrence.get(subject[-1][0],set()):
                                clusetr.append(role)
                    event_role_cluster.append(clusetr)
            else:
                event_role_cluster.append(roles)
            
            
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
    
    def predict(self,text):
        self.item_events={}
        item={"id":"single","title":"","text":text}
        dataloader = self.processor.create_dataloader([item], batch_size=1, shuffle=False)
        batch=next(iter(dataloader))
        for i in range(len(batch)-1):
            batch[i] = batch[i].to(self.device)
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self.model(input_ids, token_type_ids, attention_mask)
        preds=self.model.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)

        offset,pred_role=offset_mapping[0],preds[0]
        self.extract_events(item,offset,pred_role[0])
        events_dict=self.item_events.get(item["id"],{})
        events=self.cluster_events(events_dict,concurrence[0])
        print(events)
        return events
    
    def generate_k_temp(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ckpt_name=["val_total_f1=1.491_pf1=0.729cf1=0.762_epoch=12.ckpt",
                   "val_total_f1=1.497_pf1=0.728cf1=0.769_epoch=11.ckpt",
                   "val_total_f1=1.497_pf1=0.723cf1=0.774_epoch=10.ckpt",
                   "val_total_f1=1.475_pf1=0.715cf1=0.761_epoch=9.ckpt",
                   "val_total_f1=1.462_pf1=0.714cf1=0.747_epoch=11.ckpt"]

        all_offset_maping=[]
        test_data = self.processor.get_test_data()[30000:]
        dataloader = self.processor.create_dataloader(test_data, batch_size=self.config.batch_size, shuffle=False,is_test=True)

        for kno in range(self.config.k):
            model = NERModel.load_from_checkpoint(os.path.join(self.config.ner_save_path,f"k{kno}",ckpt_name[kno]), config=self.config)
            print("The TEST num is:", len(test_data))
            model.to(device)
            model.eval()


            all_pointer_pres=[]
            all_concurrence_pres=[]
            for batch in tqdm.tqdm(dataloader):
                for i in range(len(batch)-1):
                    batch[i] = batch[i].to(device)

                input_ids, token_type_ids, attention_mask,offset_mapping,label_tensors,sec_tensor,label_index = batch
                if kno==0:
                    all_offset_maping.append(offset_mapping.cpu().detach())
                pointer,concurrence = model(input_ids, token_type_ids, attention_mask)
                pointer=pointer.cpu().detach()
                concurrence=concurrence.cpu().detach()
                all_pointer_pres.append(pointer)
                all_concurrence_pres.append(concurrence)
            all_pointer_pres_tensor=torch.cat(all_pointer_pres,dim=0)
            all_concurrence_pres_tensor=torch.cat(all_concurrence_pres,dim=0)
            torch.save(all_pointer_pres_tensor,os.path.join("ktemp",f"pointer_{kno}.pk"))
            torch.save(all_concurrence_pres_tensor,os.path.join("ktemp",f"concurrence_{kno}.pk"))

            print('done--all %d tokens.' % len(test_data))

        if all_offset_maping:        
            all_offset_mapping_tensor=torch.cat(all_offset_maping,dim=0)
            torch.save(all_offset_mapping_tensor,os.path.join("ktemp","offset_mapping.pk"))
        
    def generate_k_result(self,outfile_txt):
        integrate_pointer_tensor=None
        integrate_concurrence_tensor=None
        kpno,kcno=0,0
        for temp_tensor_file_name in os.listdir("ktemp"):
            if temp_tensor_file_name.startswith("pointer"):
                with open(os.path.join("ktemp",temp_tensor_file_name),"rb") as fin:
                    temp_tensor=torch.load(fin)
                    if kpno==0:
                        integrate_pointer_tensor=temp_tensor
                    else:
                        integrate_pointer_tensor+=temp_tensor
                kpno+=1
            if temp_tensor_file_name.startswith("concurrence"):
                temp_tensor=torch.load(os.path.join("ktemp",temp_tensor_file_name))
                if kcno==0:
                    integrate_concurrence_tensor=temp_tensor
                else:
                    integrate_concurrence_tensor+=temp_tensor
                kcno+=1
        integrate_pointer_tensor/=kpno
        integrate_concurrence_tensor/=kcno
        
        offset_mapping=torch.load(os.path.join("ktemp","offset_mapping.pk"))
        
        test_data = self.processor.get_test_data()[30000:]

        concurrence_dict={}
        for offset,pointer,concurrence,item in zip(offset_mapping,integrate_pointer_tensor,integrate_concurrence_tensor,test_data):
            pred=self.processor.from_label_tensor_to_label_index(pointer[None],concurrence[None],offset[None])[0]
            item=dict(item.items())
            self.extract_events(item,offset,pred[0])
            concurrence_dict[item["id"]]=pred[1]

                
        with open(outfile_txt, 'w') as fout:
            for item in tqdm.tqdm(test_data):
                events_dict=self.item_events.get(item["id"],{})
                events=self.cluster_events(events_dict,concurrence_dict[item["id"]])
                item=dict(item.items())
                item["event_list"]=events
                fout.write(json.dumps(item,ensure_ascii=False)+"\n")

        print('done--all %d tokens.' % len(test_data))
