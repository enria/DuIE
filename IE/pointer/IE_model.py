# coding=utf-8
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
import os
import numpy as np
import re
import json
import time
from evaluation import evaluate_pointer,evaluate_concurrence,evaluate_span
from evaluation import F1Counter
from IE_data_util import NERProcessor,EventSchemaDict


class PointerNet(torch.nn.Module):
    def __init__(self,config,label_categories_num):
        super(PointerNet, self).__init__()
        self.encoder=BertModel.from_pretrained(config.pretrained_path)

        self.cls_dense=torch.nn.Linear(self.encoder.config.hidden_size,1)

        # Pointer Network: one label(event role) has both start and end pointer
        self.dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)
        
        # 共现矩阵 (w,h)*(h,h)->(w,h), (w,h)*(h,w)->(w,w)
        # self.dense2=torch.nn.Linear(self.encoder.config.hidden_size,self.encoder.config.hidden_size)
        self.dense3=torch.nn.Linear(self.encoder.config.hidden_size,256)
        # self.dense4=torch.nn.Linear(self.encoder.config.hidden_size,self.encoder.config.hidden_size)
        self.dense5=torch.nn.Linear(self.encoder.config.hidden_size,256)
        # self.dense6=torch.nn.Linear(self.encoder.config.hidden_size,self.encoder.config.hidden_size)

        # Binary classification: is the pointer?
        self.activation=torch.sigmoid
        self.dropout=torch.nn.Dropout(config.dropout)

    def forward(self,x):
        embedding =self.encoder(**x)[0]
        dropout_embedding=self.dropout(embedding)
        # event_detector=self.activation(self.cls_dense(embedding[:,0]))
        # event_detector=event_detector.reshape(event_detector.shape[0],1,event_detector.shape[-1])
        pointer=self.activation(self.dense(dropout_embedding))

        start_embedding=embedding
        start_embedding=self.dense3(start_embedding)
        start_embedding=self.dropout(start_embedding)

        end_embedding=embedding
        end_embedding=self.dense5(end_embedding)
        end_embedding=self.dropout(end_embedding)

        concurrence=self.activation(torch.matmul(start_embedding,end_embedding.transpose(1,2)))
        return pointer,concurrence


# class PointerNet(torch.nn.Module):
#     def __init__(self,config,label_categories_num):
#         super(PointerNet, self).__init__()
#         self.encoder=BertModel.from_pretrained(config.pretrained_path)

#         # self.cls_dense=torch.nn.Linear(self.encoder.config.hidden_size,1)
#         self.dense=torch.nn.Linear(self.encoder.config.hidden_size,label_categories_num)

#         # Pointer Network: one label(event role) has both start and end pointer
#         self.cat_dense=torch.nn.Linear(self.encoder.config.hidden_size*2,256)
#         self.con_dense=torch.nn.Linear(256,1)
        
#         # Binary classification: is the pointer?
#         self.activation=torch.sigmoid

#     def forward(self,x):
#         embedding =self.encoder(**x)[0]
#         pointer=self.activation(self.dense(embedding))
#         seq_len = embedding.size()[-2]
#         concurrence=[]
#         for ind in range(seq_len):
#             head_embedding=embedding
#             tail_embedding = embedding[:, ind, :]
#             repeat_tail_embedding = tail_embedding[:, None, :].repeat(1, seq_len, 1)

#             concat_embedding=torch.cat([head_embedding,repeat_tail_embedding],dim=-1)
#             concat_embedding=torch.tanh(self.cat_dense(concat_embedding))
#             column_concurrence=self.activation(self.con_dense(concat_embedding))
#             concurrence.append(column_concurrence)
        
#         concurrence=torch.cat(concurrence,dim=2)
#         return pointer,concurrence

class NERModel(pl.LightningModule):
    def __init__(self, config,kno=-1):
        # 1. Init parameters
        super(NERModel, self).__init__()
        self.config=config
        self.kno=kno

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.use_bert = config.use_bert
        
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
        train_data = self.processor.get_train_data()
        dev_data=self.processor.get_dev_data()

        train_data_len=len(train_data)
        all_data=train_data+dev_data

        import random
        random.seed()
        random.shuffle(all_data)
        train_data=all_data[:train_data_len]
        dev_data=all_data[train_data_len:]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_dataset = self.processor.create_dataset(train_data)
        self.valid_dataset = self.processor.create_dataset(dev_data)
    
    def prepare_k_data(self):
        train_data = self.processor.get_train_data()
        dev_data=self.processor.get_dev_data()
        import random
        random.shuffle(train_data)

        all_data=dev_data+train_data
        all_data_num=len(all_data)
        k_avg_num=len(dev_data)
        import math

        k_fold_train_data,k_fold_dev_data=[],None
        for i in range(math.ceil(all_data_num/k_avg_num)):
            cur_data_split=all_data[i*k_avg_num:(i+1)*k_avg_num]
            if i==self.kno:
                k_fold_dev_data=cur_data_split
            else:
                k_fold_train_data.extend(cur_data_split)
        
        print("train_length:", len(k_fold_train_data))
        print("valid_length:", len(k_fold_dev_data))

        self.train_dataset = self.processor.create_dataset(k_fold_train_data)
        self.valid_dataset = self.processor.create_dataset(k_fold_dev_data)

    def train_dataloader(self):
        random_sampler=torch.utils.data.RandomSampler(self.train_dataset,replacement=True,num_samples=self.config.train_sample_size)
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=self.train_dataset.collate_fn,
            sampler=(random_sampler if self.config.sample else None)
        )
        return dataloader

    def val_dataloader(self):
        random_sampler=torch.utils.data.RandomSampler(self.valid_dataset,replacement=True,num_samples=self.config.dev_sample_size)
        dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=self.valid_dataset.collate_fn,
            sampler=(random_sampler if self.config.sample else None)
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0], label_tensors[attention_mask!=0])
        # pointer_loss=torch.tensor(0)
        concurrence_loss= self.criterion(concurrence[attention_mask!=0], concurrence_tensors[attention_mask!=0])
        loss=pointer_loss+concurrence_loss

        self.log('pointer_loss', pointer_loss.item(),prog_bar=True)
        self.log('concurrence_loss', concurrence_loss.item(),prog_bar=True)
        self.log('train_loss', loss.item())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
        pointer,concurrence = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pointer_loss = self.criterion(pointer[attention_mask!=0], label_tensors[attention_mask!=0])
        # pointer_loss=torch.tensor(0)
        concurrence_loss= self.criterion(concurrence[attention_mask!=0], concurrence_tensors[attention_mask!=0])
        loss=pointer_loss+concurrence_loss

        span_pred=self.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)
        span_pred,_,_=zip(*span_pred)

        pointer_counter=F1Counter()
        evaluate_pointer(torch.where(pointer[attention_mask!=0]>0.5,1,0),label_tensors[attention_mask!=0],pointer_counter)

        span_counter=F1Counter()
        evaluate_span(span_pred,label_text_index,span_counter)

        concurrence_counter=F1Counter()
        evaluate_concurrence(torch.where(concurrence[attention_mask!=0]>0.5,1,0),concurrence_tensors[attention_mask!=0],concurrence_counter)

        return pointer_loss,concurrence_loss,pointer_counter,span_counter,concurrence_counter


    def validation_epoch_end(self, outputs):
        pointer_total_counter=F1Counter()
        span_total_counter=F1Counter()
        concurrence_total_counter=F1Counter()
        pointer_totol_losss,concurrence_total_loss=0,0
        for output in outputs:
            pointer_loss,concurrence_loss,pointer_counter,span_counter,concurrence_counter = output
            pointer_totol_losss+=pointer_loss.item()
            concurrence_total_loss+=concurrence_loss.item()
            pointer_total_counter+=pointer_counter
            span_total_counter+=span_counter
            concurrence_total_counter+=concurrence_counter

        pointer_precision,pointer_recall,pointer_f1=pointer_total_counter.cal_score()
        concurrence_precision,concurrence_recall,concurrence_f1=concurrence_total_counter.cal_score()

        self.log('pf1', torch.tensor(pointer_f1))
        self.log('cf1', torch.tensor(concurrence_f1))
        self.log('val_total_f1', pointer_f1+concurrence_f1)

        print(f"\nFinished epoch ploss:{pointer_totol_losss:.4f}, pprec:{pointer_precision:.4f}, precall:{pointer_recall:.4f}, pf1:{pointer_f1:.4f}\t"+
              f"closs:{concurrence_total_loss:.4f}, cprec:{concurrence_precision:.4f}, crecall:{concurrence_recall:.4f}, cf1:{concurrence_f1:.4f}")


class NERPredictor:
    def __init__(self, checkpoint_path, config,):
        self.use_bert = config.use_bert
        self.config=config
        self.checkpoint_path=checkpoint_path

        # self.model = NERModel.load_from_checkpoint(checkpoint_path, config=config)

        # self.test_data = self.model.processor.get_test_data()

        # self.tokenizer = self.model.tokenizer
        # self.dataloader = self.model.processor.create_dataloader(
        #     self.test_data, batch_size=config.batch_size, shuffle=False)
        
        self.event_schema=EventSchemaDict(config.schema_path)
        self.processor=NERProcessor(config)

        # print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

        self.role_total_num,self.use_role_num=0,0


    def extract_events(self,item,offset_mapping,preds):
        events_dict={}
        for index,label_id in preds:
            argument=item["text"][index[0]:index[1]]
            if not argument:
                continue
            label=self.event_schema.id2labels[label_id[0]]
            event_type,role=re.match("B-(.+):(.+)",label).groups()
            events_dict.setdefault(event_type,set())
            if argument:
                events_dict[event_type].add((role,argument,index))
        return events_dict

    def cluster_events(self,events_dict,concurrence,relax_concurrence,fspy):
        new_events=[]
        for event_type,roles in events_dict.items():
            subjects=[]
            objects=[]
            assist_roles={}

            for role in roles:
                if role[0] =="subject":
                    subjects.append(role)
                elif role[0] == "@value":
                    objects.append(role)
                else:
                    assist_roles.setdefault(role[0],[])
                    assist_roles[role[0]].append(role)

            SO_pairs=[]
            event_role_cluster=[]
            if len(subjects)==1 and len(objects)==1:
                SO_pairs=[(subjects[0],objects[0])]
                clusetr=[subjects[0],objects[0]]
                for role_type,roles in assist_roles.items():
                    for role in roles:
                        clusetr.append(role)
                event_role_cluster.append(clusetr)
            else:
                remain_subject_set=set(subjects)
                remain_object_set=set(objects)
                for s in subjects:
                    for o in objects:
                        if o[-1][0] in concurrence.get(s[-1][0],set()):
                            SO_pairs.append((s,o))
                            if s in remain_subject_set:remain_subject_set.remove(s)
                            if o in remain_object_set:remain_object_set.remove(o)
                
                if remain_subject_set:
                    for s in remain_subject_set:
                        rco=list(filter(lambda o:o[-1][0] in relax_concurrence.get(s[-1][0],{}),objects))
                        if rco:
                            mrco=max(rco,key=lambda o:relax_concurrence[s[-1][0]][o[-1][0]])
                            SO_pairs.append((s,mrco))
                            if mrco in remain_object_set:remain_object_set.remove(mrco)
                        else:
                            SO_pairs.append((s,))
                
                if remain_object_set:
                    for o in remain_object_set:
                        rcs=list(filter(lambda s:s[-1][0] in relax_concurrence.get(o[-1][0],{}), subjects))
                        if rcs:
                            mrcs=max(rcs,key=lambda s:relax_concurrence[o[-1][0]][s[-1][0]])
                            SO_pairs.append((mrcs,o))
                        else:
                            SO_pairs.append((o,))

                if remain_subject_set or remain_object_set:
                    fspy.write(f"remain {event_type}: {remain_subject_set} {remain_object_set}\n")
                        
                for sao in SO_pairs:
                    if len(sao)>=2:
                        clusetr=[sao[0],sao[1]]
                        for role_type,roles in assist_roles.items():
                            for role in roles:
                                if role[-1][0] in concurrence.get(sao[0][-1][0],set()) \
                                or role[-1][0] in concurrence.get(sao[1][-1][0],set()):
                                    clusetr.append(role)
                    else:
                        continue
                    event_role_cluster.append(clusetr)
            
            self.role_total_num+=len(subjects)+len(objects)
            self.use_role_num+=2*len(SO_pairs)

            
            for role_cluster in event_role_cluster:
                spo={"predicate":event_type,"object":{}}
                for role in role_cluster:
                    if role[0]=="subject":
                        spo["subject"]=role[1]
                    else:
                        spo["object"][role[0]]=role[1]
                new_events.append(spo)
        return new_events

    def generate_result(self, outfile_txt):
        self.model = NERModel.load_from_checkpoint(self.checkpoint_path, config=self.config)
        

        self.test_data = self.model.processor.get_test_data()
        dataset = self.processor.create_dataset(self.test_data)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=dataset.collate_fn
        )

        
        self.event_schema=EventSchemaDict(self.config.schema_path)
        self.processor=NERProcessor(self.config)

        # print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', self.checkpoint_path)

        self.role_total_num,self.use_role_num=0,0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0
        concurrence_dict={}
        with open(outfile_txt, 'w') as fout,open("spy_result.json", 'w') as fspy:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)-1):
                    batch[i] = batch[i].to(device)

                input_ids, token_type_ids, attention_mask,offset_mapping,label_tensors,concurrence_tensors,label_text_index = batch

                pointer,concurrence = self.model(input_ids, token_type_ids, attention_mask)
                preds=self.model.processor.from_label_tensor_to_label_index(pointer,concurrence,offset_mapping)

                for offset,pred_role in zip(offset_mapping,preds):
                    item=dict(self.test_data[cnt].items())
                    fspy.write(f"{item['text']}\n")
                    events_dict=self.extract_events(item,offset,pred_role[0])
                    events=self.cluster_events(events_dict,pred_role[1],pred_role[2],fspy)
                    item=dict(item.items())
                    item["spo_list"]=events
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")
                    cnt+=1
                    fspy.write(f"{events}\n\n")
        
        print('done--all %d tokens.' % cnt)
        print(self.use_role_num,self.role_total_num)
    
    def generate_k_temp(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ckpt_name=["val_f1=1.6492_epoch=4_dropout+detach.ckpt",
                   "val_total_f1=1.634_pf1=0.810cf1=0.824_epoch=3.ckpt",
                   "val_total_f1=1.624_pf1=0.808cf1=0.816_epoch=3.ckpt"]

        all_offset_maping=[]
        test_data = self.processor.get_test_data()[50000:]
        dataset = self.processor.create_dataset(test_data)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=dataset.collate_fn
        )

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
        
        with open(os.path.join("ktemp","offset_mapping.pk"),"rb") as fin:
            offset_mapping=torch.load(fin)
        
        test_data = self.processor.get_test_data()[50000:]

        with open(outfile_txt, 'w') as fout,open("spy_result.json", 'w') as fspy:
            for offset,pointer,concurrence,item in zip(offset_mapping,integrate_pointer_tensor,integrate_concurrence_tensor,test_data):
                pred=self.processor.from_label_tensor_to_label_index(pointer[None],concurrence[None],offset[None])[0]
                item=dict(item.items())
                events_dict=self.extract_events(item,offset,pred[0])
                events=self.cluster_events(events_dict,pred[1],pred[2],fspy)
                item["spo_list"]=events
                fout.write(json.dumps(item,ensure_ascii=False)+"\n")

        print('done--all %d tokens.' % len(test_data))

