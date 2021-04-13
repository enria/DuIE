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

error_cnt=0
label_error_cnt=0
global_text_index=0

class Dataset(torch.utils.data.Dataset):
    def __init__(self,input_ids,token_type_ids,attention_mask,offset_mapping,label_tensor,label_text_index):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.offset_mapping=offset_mapping
        self.label_tensor=label_tensor
        self.label_text_index=label_text_index

    def __getitem__(self, index):
        return (self.input_ids[index],
               self.token_type_ids[index],
               self.attention_mask[index],
               self.offset_mapping[index],
               self.label_tensor[index],
               self.label_text_index[index])

    def __len__(self):
        return self.input_ids.size(0)

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        for i in range(len(batch)-1):
            batch[i]=torch.stack(batch[i])
        return batch

class NERProcessor(DataProcessor):
    """
       从NER数据文件/data/NER_data下读取数据，生成NER训练数据dataloader，返回给模型
    """

    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.test_path=config.test_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.event_schema=EventSchemaDict(config.schema_path)

    def read_json_data(self, filename):
        data = []
        with open(filename, 'r') as fjson:
            for line in fjson:
                item = json.loads(line)
                data.append(item)
        return data

    def get_train_data(self):
        data = self.read_json_data(self.train_path)
        return data
    
    def get_dev_data(self):
        data = self.read_json_data(self.dev_path)
        return data
    
    def get_test_data(self):
        test_data = self.read_json_data(self.test_path)
        return test_data

    def from_offset_to_label_tensor(self,offset,label_index):
        label_tensor=torch.zeros(len(offset),len(self.event_schema.id2labels))
        offset_index_dict={} # text index to offset index
        for i,o in enumerate(offset):
            if o[-1]==0:
                continue
            for oi in range(o[0],o[-1]):
                offset_index_dict[int(oi)]=i

        for role_index in label_index:
            try:
                start=offset_index_dict[role_index[0][0]]
                end=offset_index_dict[role_index[0][1]-1]
                label_tensor[start][role_index[1][0]]=1
                label_tensor[end][role_index[1][1]]=1
            except:
                global error_cnt
                error_cnt+=1
                print("match: %d"%global_text_index)

        return label_tensor

    def create_dataloader(self, data, batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer

        # 2. 分别对句子和公司进行编码表示
        text = [d["text"] for d in data]
        max_length = min(max_length, max([len(s) for s in text]))
        print("max sentence length: ", max_length)

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        label_tensors = []
        label_text_index=[]
        for index,offset in enumerate(inputs['offset_mapping']):
            global global_text_index
            global_text_index=index
            event_list=data[index].get("event_list",[])
            label_index=self.event_schema.tokens_to_label_index(data[index]["text"],event_list)
            label_tensor=self.from_offset_to_label_tensor(offset,label_index)
            label_tensors.append(label_tensor)
            label_text_index.append([(x[0],x[1]) for x in label_index])

        # dataset = torch.utils.data.TensorDataset(
        #     torch.LongTensor(inputs["input_ids"]),          # 句子字符id
        #     torch.LongTensor(inputs["token_type_ids"]),     # 区分两句话，此任务中全为0，表示输入只有一句话
        #     torch.LongTensor(inputs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0
        #     torch.LongTensor(inputs["offset_mapping"]),     # 区分是否是pad值。句子内容为1，pad为0
        #     torch.stack(labels)
        # )

        dataset=Dataset(inputs["input_ids"],inputs["token_type_ids"],inputs["attention_mask"],inputs["offset_mapping"],label_tensors,label_text_index)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=Dataset.collate_fn
        )
        return dataloader
    
    def from_label_tensor_to_label_index(self,label_tensor_list,offset_mapping_list,threshold=0.5):
        result=[]
        for label_tensor,offset_mapping in zip(label_tensor_list,offset_mapping_list):
            label_index=(label_tensor>threshold).nonzero()
            label_text_index={} # key:label_id,value:text indices

            item_result=[]
    
            for index in label_index:
                key=index[1].item()
                row=index[0].item()
                if offset_mapping[row][-1]==0:
                    continue
                label_text_index.setdefault(key,[])
                label_text_index[key].append(row)
            
            for start_label_id,start_text_indices in label_text_index.items():
                if start_label_id%2:
                    continue
                end_label_id=start_label_id+1
                end_text_indices=label_text_index.get(end_label_id,[])
                start_index,end_index=0,0
                while start_index<len(start_text_indices) and end_index<len(end_text_indices):
                    while end_index<len(end_text_indices) and end_text_indices[end_index]<start_text_indices[start_index]:
                        end_index+=1
                    if end_index>=len(end_text_indices):
                        break
                    item_result.append(((offset_mapping[start_text_indices[start_index]][0].item(),
                                    offset_mapping[end_text_indices[end_index]][-1].item()),
                                    (start_label_id,end_label_id)))
                    start_index+=1
            result.append(item_result)

        return result

class EventSchemaDict(object):
    def __init__(self,schema_path,use_trigger=True):
        self.schema_path=schema_path
        self.use_trigger=use_trigger
        self.label2ids={}
        self.id2labels={}

        self.load_schema()

    def load_schema(self):
        role_list=[]
        with open(self.schema_path) as schema:
            for line in schema:
                if not line:
                    continue
                event_schema=json.loads(line)
                event_type=event_schema["event_type"]

                if self.use_trigger:
                    #添加事件触发词
                    role_list.append(event_type+":"+"TRG") 

                for role in event_schema["role_list"]:
                    role_list.append(event_type+":"+role["role"])

        for index,role in enumerate(role_list):
            self.label2ids["B-"+role]=index*2
            self.label2ids["E-"+role]=index*2+1
            self.id2labels[index*2]="B-"+role
            self.id2labels[index*2+1]="E-"+role
    
    def tokens_to_label_index(self,text:str,label):
        label_index=[]
        for event_index,event in enumerate(label):
            event_type=event["event_type"]

            addition_roles=[]
            if self.use_trigger:
                trigger={"role":"TRG",
                         "argument":event["trigger"],
                         "argument_start_index":event["trigger_start_index"]}
                addition_roles.append(trigger)

            for role in event["arguments"]+addition_roles:
                role_start=int(role["argument_start_index"])
                role_end=role_start+len(role["argument"])

                # 有些数据标注错误，加上了空格。用BertTokenizer不好处理，而且这样的数据会是噪声。
                if role["argument"]!=role["argument"].strip():
                    if role["argument"].strip() in role["argument"]:
                        role_start+=role["argument"].index(role["argument"].strip())
                        role_end=role_start+len(role["argument"].strip())
                B_label_id=self.label2ids["B-"+event_type+":"+role["role"]]
                I_label_id=self.label2ids["E-"+event_type+":"+role["role"]]
                label_index.append(((role_start,role_end),(B_label_id,I_label_id),event_index))

        # Sort by argument first index and event index.
        label_index.sort(key=lambda x:(x[0][0],x[2]))
        return label_index
    
from collections import Counter
def overlap_stat(data):
    all_event,contain2event=0,0
    all_role_cnt,distinct_role_cnt=0,0
    for item in data:
        roles=[(role["argument_start_index"],len(role["argument"])) for event in item["event_list"] for role in event["arguments"]]
        if len(item["event_list"])>=2:
            contain2event+=1
            cnter=Counter(roles)
            distinct_role_cnt+=len(cnter.values())
        else:
            distinct_role_cnt+=len(roles)
        all_role_cnt+=len(roles)
        all_event+=1
    
    print(f"events:{contain2event}/{all_event}={contain2event/all_event:.2f},roles={distinct_role_cnt}/{all_event}={distinct_role_cnt/all_role_cnt:.2f}")



if __name__ == '__main__':

    from collections import namedtuple

    config_class = namedtuple('config',['train_path','dev_path','test_path','pretrained_path','schema_path'])
    config = config_class("../data/duee_train.json","../data/duee_dev.json","../data/duee_test1.json","/storage/public/models/bert-base-chinese","../data/duee_event_schema.json")

    processor=NERProcessor(config)
    train_data=processor.get_dev_data()
    loader=processor.create_dataloader(train_data,batch_size=8)
    for batch in loader:
        x = batch