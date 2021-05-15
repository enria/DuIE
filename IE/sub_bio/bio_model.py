# coding=utf-8
from collections import defaultdict
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
from torchcrf import CRF
import numpy as np
import re
import json
from conlleval import evaluate,count_chunks,ConllCounter
from IE_data_util import ObjectProcessor


class ObjectRecModule(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(ObjectRecModule, self).__init__()
        self.config=config

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.crf_lr=config.crf_lr
        self.dropout = config.dropout
        self.optimizer = config.optimizer

        self.layerNorm = torch.nn.LayerNorm

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        self.processor = ObjectProcessor(config, self.tokenizer)
        self.labels = len(self.processor.object_schema.id2labels)
        self.model = BertForTokenClassification.from_pretrained(config.pretrained_path, num_labels=self.labels)
        self.add_subject_label_token()

        self.crf = CRF(num_tags=self.labels, batch_first=True)

        self.criterion = nn.BCELoss(reduction="sum")

        print("ObjectRecModule init: done.")
    
    def add_subject_label_token(self):
        self.tokenizer.add_tokens(self.processor.object_schema.subject_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        feats = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
        )[0]

        return feats
    
    def configure_optimizers(self):
        crf_params_ids = list(map(id, self.crf.parameters()))
        base_params = filter(lambda p: id(p) not in crf_params_ids, [
                                 p for p in self.parameters() if p.requires_grad])
        arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]

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

        self.train_set = self.processor.create_dataset(train_data)
        self.valid_set = self.processor.create_dataset(dev_data)

    def train_dataloader(self):
        random_sampler=torch.utils.data.RandomSampler(self.train_set,replacement=True,num_samples=20000)
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=self.train_set.collate_fn,
            sampler=random_sampler
        )

    def val_dataloader(self):
        random_sampler=torch.utils.data.RandomSampler(self.valid_set,replacement=True,num_samples=4000)
        return torch.utils.data.DataLoader(
            self.valid_set,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=4,
            collate_fn=self.valid_set.collate_fn,
            sampler=random_sampler
        )
    
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, labels,subject_labe_texts,line_no = batch
        feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = -self.crf(feats, labels,mask=attention_mask.byte(), reduction='mean')

        self.log('train_loss', loss.item())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask,offset_mapping, labels,subject_labe_texts,line_no = batch
        feats = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        loss = -self.crf(feats, labels,mask=attention_mask.byte(), reduction='mean')

        pred = self.crf.decode(feats)
        gold = labels[attention_mask != 0].tolist()
        pre = torch.tensor(pred).cuda()[attention_mask != 0].tolist()

        true_seqs = [self.processor.object_schema.id2labels[int(g)] for g in gold]
        pred_seqs = [self.processor.object_schema.id2labels[int(g)] for g in pre]
        counter=count_chunks(true_seqs,pred_seqs,ConllCounter())

        return {"loss":loss,"counter":counter}

    def validation_epoch_end(self, outputs):
        
        val_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        counters=[x['counter'] for x in outputs]

        print('\n')
        prec, rec, f1 = evaluate(counters)
        f1=f1/100

        self.log('val_loss', val_loss)
        self.log('val_pre', torch.tensor(prec))
        self.log('val_rec', rec)
        self.log('val_f1', torch.tensor(f1))

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
          return list(obj)
        return json.JSONEncoder.default(self, obj)


class ObjectPredictor:
    def __init__(self, checkpoint_path, config):

        self.model = ObjectRecModule.load_from_checkpoint(checkpoint_path, config=config)
        self.config=config

        self.test_data = self.model.processor.get_test_data()
        self.tokenizer = self.model.tokenizer
        self.test_dataset = self.model.processor.create_dataset(self.test_data)

        self.dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=config.batch_size,
            num_workers=4,
            collate_fn=self.test_dataset.collate_fn,
        )

        self.object_schema=self.model.processor.object_schema

        print("The TEST num is:", len(self.test_dataset))
        print('load checkpoint:', checkpoint_path)

    def extract_objects(self,text,offset_mapping,preds):
        subject_match=re.findall("<s-(\d+)>(.+?)</s-\d+>",text)
        if not subject_match:
            return []
        subject_match=subject_match[0]
        subject_predicate=self.object_schema.predicates[int(subject_match[0])]
        subject_mention=subject_match[1]
        if not subject_mention.strip():return []

        spo_dict={}
        index=1 #直接跳过CLS
        last_B,last_B_index=0,0
        while index<len(preds):
            if last_B:
                argument=None
                if preds[index]!=last_B+1 or offset_mapping[index][1]==0:
                    argument=text[offset_mapping[last_B_index][0]:offset_mapping[index-1][1]]
                elif index==len(preds)-1:
                    argument=text[offset_mapping[last_B_index][0]:offset_mapping[index][1]]
                if argument:
                    label=self.model.processor.object_schema.id2labels[last_B]
                    label=re.match("B-(.+):(.+)",label).groups()
                    spo_dict.setdefault(label[0],[])
                    if argument:
                        spo_dict[label[0]].append((label[1],argument,last_B_index))
                    last_B,last_B_index=0,0

            if preds[index]!=0 and preds[index]%2:
                last_B,last_B_index=preds[index],index
            index+=1
        spos=[]
        if subject_predicate in spo_dict:
            roles=spo_dict[subject_predicate]
            if subject_predicate in self.object_schema.complicated_spo:
                values=[item for item in roles if item[0]=="@value"]
                assists=[item for item in roles if item[0]!="@value"]
                for value in values:
                    value_assist={}
                    for assist in assists:
                        if assist[0] not in  value_assist or \
                            abs(assist[-1]-value[-1])<abs(value_assist[assist[0]][-1]-value[-1]):
                            value_assist[assist[0]]=assist
                    for key in value_assist:
                        value_assist[key]=value_assist[key][1]
                    value_assist["@value"]=value[1]
                    spos.append({
                        "subject":subject_mention,
                        "predicate":subject_predicate,
                        "object":value_assist
                    })

            else:
                for role in roles:
                    spos.append({
                        "subject":subject_mention,
                        "predicate":subject_predicate,
                        "object":{
                            "@value":role[1]
                        }
                    })
        return spos


    def generate_result(self, outfile_txt):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        cnt = 0

        line_spo_dict=defaultdict(list)

        
        for batch in tqdm.tqdm(self.dataloader):
            for i in range(len(batch)-2):
                batch[i] = batch[i].to(device)
            input_ids, token_type_ids, attention_mask,offset_mapping,labels,subject_labe_texts,line_no = batch

            emissions = self.model(input_ids, token_type_ids, attention_mask)
            preds = self.model.crf.decode(emissions)  # list

            for offset,pred,slt,no in zip(offset_mapping,preds,subject_labe_texts,line_no):
                spos=self.extract_objects(slt,offset,pred)
                line_spo_dict[no].extend(spos)
        
        with open(outfile_txt, 'w') as fout:
            for no,item in enumerate(self.test_data):
                item=dict(item.items())
                item["spo_list"]=line_spo_dict[no]
                fout.write(json.dumps(item,ensure_ascii=False)+"\n")

        print('done--all %d tokens.' % cnt)
    