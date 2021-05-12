import sys
import os

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from transformers import (
    DataProcessor,
    BertTokenizerFast
)
from collections import defaultdict
import torch
import csv
import numpy as np
import json
import pytorch_lightning as pl

error_cnt=0
label_error_cnt=0
global_text_index=0
miss_cnt=0

class Dataset(torch.utils.data.Dataset):
    def __init__(self,input_ids,token_type_ids,attention_mask,label_tensor):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.label_tensor=label_tensor

    def __getitem__(self, index):
        return (self.input_ids[index],
               self.token_type_ids[index],
               self.attention_mask[index],
               self.label_tensor[index])

    def __len__(self):
        return self.input_ids.size(0)

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        for i in range(len(batch)):
            batch[i]=torch.stack(batch[i])
        return batch


class IPOProcessor(DataProcessor):
    """
       从NER数据文件/data/NER_data下读取数据，生成NER训练数据dataloader，返回给模型
    """

    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.IPO_categories=[
                "筹备上市",
                "暂停上市",
                "正式上市",
                "终止上市"
            ]

    def read_json_data(self, filename, add_title=False):
        data = []
        with open(filename, 'r') as fjson:
            for line in fjson:
                item = json.loads(line)
                for event in item.get("event_list",[]):
                    if event["event_type"]=="公司上市":
                        if add_title:
                            item=self.add_title(item)
                        data.append(item)
                        break
                    
        return data

    def get_train_data(self,add_title=True):
        data = self.read_json_data(self.train_path,add_title)
        return data
    
    def get_dev_data(self,add_title=True):
        data = self.read_json_data(self.dev_path,add_title)
        return data
    
    def get_test_data(self,test_path,add_title=True):
        test_data = self.read_json_data(test_path,add_title)
        return test_data
    
    # 确保标题出现在文本中
    @staticmethod
    def add_title(item):
        if item["text"].find(item["title"],0,len(item["title"])+20)<0:
            item["text"]=f'原标题：{item["title"]}    {item["text"]}'
        return item
    
    def get_IPO_cls(self,item):
        for event in item["event_list"]:
            if event["event_type"]=="公司上市":
                for role in event["arguments"]:
                    if role["role"]=="环节":
                        return role["argument"]

        return "筹备上市"
    
    def create_dataloader(self, data, batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer

        text = [d["text"] for d in data]
        
        max_length = min(max_length, max([len(s) for s in text]))
        print("max sentence length: ", max_length)

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )

        label_tensors = []
        concurrence_tensors=[]
        label_text_index=[]
        print(len(data))

        for index in range(len(data)):
            IPO_cls=self.get_IPO_cls(data[index])
            label_index=self.IPO_categories.index(IPO_cls) if IPO_cls in self.IPO_categories else -1
            label_tensors.append(torch.tensor(label_index))


        dataset=Dataset(inputs["input_ids"],
                        inputs["token_type_ids"],
                        inputs["attention_mask"],
                        label_tensors)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=Dataset.collate_fn
        )
        return dataloader
    
if __name__ == '__main__':
    # schema=EventSchemaDict("../data/duee_fin_event_schema_sub.json")
    # print(schema.event_subject['股东减持'])
    # exit() 

    from collections import namedtuple

    config_class = namedtuple('config',['train_path','dev_path','test_path','pretrained_path','schema_path'])
    config = config_class("../data/duee_fin_train.json","../data/duee_fin_dev.json","../data/duee_fin_test1.json","/storage/public/models/bert-base-chinese","../data/duee_fin_event_schema_sub.json")

    processor=IPOProcessor(config)
    train_data=processor.get_train_data()
    for b in processor.create_dataloader(train_data,8):
        b
