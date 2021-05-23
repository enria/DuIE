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
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self,input_ids,token_type_ids,attention_mask,offset_mapping,label_tensor,concurrence_tensor,label_text_index):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.offset_mapping=offset_mapping
        self.label_tensor=label_tensor
        self.concurrence_tensor=concurrence_tensor
        self.label_text_index=label_text_index

    def __getitem__(self, index):
        return (self.input_ids[index],
               self.token_type_ids[index],
               self.attention_mask[index],
               self.offset_mapping[index],
               self.label_tensor[index],
               self.concurrence_tensor[index],
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
    
    def from_offset_to_label_tensor(self,offset,label_index,offset_index_dict):
        label_tensor=torch.zeros(len(offset),len(self.event_schema.id2labels))

        for role_index in label_index:
            if role_index[0][0]>512 or\
               role_index[0][0] not in offset_index_dict or \
               role_index[0][1]-1 not in offset_index_dict:
                print("missing")
                continue
            start=offset_index_dict[role_index[0][0]]
            end=offset_index_dict[role_index[0][1]-1]
            label_tensor[start][role_index[1][0]]=1
            label_tensor[end][role_index[1][1]]=1

        return label_tensor
    

    def create_dataset(self, data, max_length=512):
        tokenizer = self.tokenizer

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
        concurrence_tensors=[]
        label_text_index=[]
        for index,offset in tqdm(enumerate(inputs['offset_mapping']),total=len(data)):
            offset_index_dict={} # text index to offset index
            for i,o in enumerate(offset):
                if o[-1]==0:
                    continue
                for oi in range(o[0],o[-1]):
                    offset_index_dict[int(oi)]=i
            event_list=data[index].get("spo_list",[])
            concurrence_matrix=torch.zeros((offset.shape[0],offset.shape[0]))
            label_index=self.event_schema.tokens_to_label_index(text[index],event_list,offset_index_dict,concurrence_matrix)
            label_tensor=self.from_offset_to_label_tensor(offset,label_index,offset_index_dict)
            label_tensors.append(label_tensor)
            concurrence_tensors.append(concurrence_matrix)
            label_text_index.append([(x[0],x[1]) for x in label_index])


        dataset=Dataset(inputs["input_ids"],
                        inputs["token_type_ids"],
                        inputs["attention_mask"],
                        inputs["offset_mapping"],
                        label_tensors,
                        concurrence_tensors,
                        label_text_index)

        return dataset
    
    def from_label_tensor_to_label_index(self,label_tensor_list,concurrence_tensor_list,offset_mapping_list,threshold=0.5):
        result=[]
        for index in range(len(label_tensor_list)):
            label_tensor,concurrence_tensor,offset_mapping=label_tensor_list[index],concurrence_tensor_list[index],offset_mapping_list[index]
            label_index=(label_tensor>0.3).nonzero()
            concurrence_index=((concurrence_tensor+concurrence_tensor.transpose(0,1))>threshold*2).nonzero()
            label_text_index={} # key:label_id,value:text indices

            item_result=[]
            concurrence={}
    
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
                    start_input_index = start_text_indices[start_index]
                    while end_index<len(end_text_indices) and end_text_indices[end_index]<start_text_indices[start_index]:
                        end_index+=1
                    if end_index>=len(end_text_indices):
                        break
                    
                    end_input_index=end_text_indices[end_index]
                    if start_index<len(start_text_indices)-1:
                        if start_text_indices[start_index+1]<end_text_indices[end_index]:
                            end_input_index=-1

                    if end_input_index!=-1:
                        item_result.append(((offset_mapping[start_input_index][0].item(),
                                            offset_mapping[end_input_index][-1].item()),
                                            (start_label_id,end_label_id)))
                    start_index+=1
            
            for index in concurrence_index:
                key=index[1].item()
                row=index[0].item()
                if offset_mapping[row][-1]==0 or offset_mapping[key][-1]==0 :
                    continue
                concurrence.setdefault(offset_mapping[key][0].item(),set())
                concurrence[offset_mapping[key][0].item()].add(offset_mapping[row][0].item())

            relax_concurrence={}
            relax_concurrence_tensor=(concurrence_tensor+concurrence_tensor.transpose(0,1))>0.3
            relax_concurrence_index=relax_concurrence_tensor.nonzero()
            for index in relax_concurrence_index:
                key=index[1].item()
                row=index[0].item()
                if offset_mapping[row][-1]==0 or offset_mapping[key][-1]==0 :
                    continue
                relax_concurrence.setdefault(offset_mapping[key][0].item(),{})
                relax_concurrence[offset_mapping[key][0].item()][offset_mapping[row][0].item()]=relax_concurrence_tensor[row][key]

            result.append((item_result,concurrence,relax_concurrence))

        return result

class EventSchemaDict(object):
    def __init__(self,schema_path,use_trigger=True,find_closest=False):
        self.schema_path=schema_path
        self.use_trigger=use_trigger
        self.find_closest=find_closest
        self.label2ids={}
        self.id2labels={}

        self.complicated_event_type=set()

        self.load_schema()

    def load_schema(self):
        role_list=[]
        with open(self.schema_path) as schema:
            for line in schema:
                if not line:
                    continue
                event_schema=json.loads(line)
                event_type=event_schema["predicate"]

                role_list.append(f"{event_type}:subject")
                for key, desc in event_schema["object_type"].items():
                    role_list.append(f"{event_type}:{key}")
                if len(event_schema["object_type"])>1:
                    self.complicated_event_type.add(event_type)

        for index,role in enumerate(role_list):
            self.label2ids["B-"+role]=index*2
            self.label2ids["E-"+role]=index*2+1
            self.id2labels[index*2]="B-"+role
            self.id2labels[index*2+1]="E-"+role
    
    def find_all_sub(self,a_str,sub,end=510):
        start = 0
        while True:
            start = a_str.find(sub, start,end)
            if start == -1: return []
            yield (start,start+len(sub))
            start += len(sub) 
    
    def find_closest_roles(self,role_mentions):
        """roles:{"text":{"role_label":(B,I),"mentions":[(start,end)]}}"""
        arguments=list(role_mentions.keys())
        arguments.sort(key=lambda x: len(role_mentions[x]["mentions"]))

        best_path=[]
        best_cost=-1

        def f(path,ai,cost):
            nonlocal best_cost,best_path
            if ai>=len(arguments):
                if cost<best_cost or best_cost==-1:
                    best_path=path[:]
                    best_cost=cost
                return 

            mentions=role_mentions[arguments[ai]]["mentions"]
            for mi in range(len(mentions)):
                ncost=cost
                for aj in range(len(path)):
                    ncost+=abs(mentions[mi][0]-role_mentions[arguments[aj]]["mentions"][path[aj]][0])
                if best_cost==-1 or ncost<best_cost:
                    f(path+[mi],ai+1,ncost)

        f([],0,0)
        select_arguments=[(role_mentions[arguments[i]]["mentions"][best_path[i]],role_mentions[arguments[i]]["role_label"]) for i in range(len(arguments))]

        return select_arguments
    
    def find_first_roles(self,roles):
        """roles:{"text":{"role_label":(B,I),"mentions":[start]}}"""

        pass

    
    def tokens_to_label_index(self,text:str,label,offset_index_dict,concurrence_matrix):
        label_index=[]
        add_index_event_list=[]
        for event_index,event in enumerate(label):
            event_type=event["predicate"]
            role_mentions={}

            mention=event["subject"].strip()
            if not mention:
                continue
            B_label_id=self.label2ids[f"B-{event_type}:subject"]
            I_label_id=self.label2ids[f"E-{event_type}:subject"]
            if self.find_closest:
                role_index=list(self.find_all_sub(text,mention))
                if len(role_index)==0: continue
                role_mentions[mention]={"role_label":(B_label_id,I_label_id),"mentions":role_index}
            else:
                role_start=text.find(mention)
                role_end=role_start+len(mention)
                if role_start<0: continue
                role_mentions[mention]={"role_label":(B_label_id,I_label_id),"mentions":[(role_start,role_end)]}

            for key,mention in event["object"].items():
                # 有些数据标注错误，加上了空格。用BertTokenizer不好处理，而且这样的数据会是噪声。
                mention=mention.strip()
                if not mention:
                    continue
                B_label_id=self.label2ids[f"B-{event_type}:{key}"]
                I_label_id=self.label2ids[f"E-{event_type}:{key}"]
                if self.find_closest:
                    role_index=list(self.find_all_sub(text,mention))
                    if len(role_index)==0: continue
                    role_mentions[mention]={"role_label":(B_label_id,I_label_id),"mentions":role_index}
                else:
                    role_start=text.find(mention)
                    role_end=role_start+len(mention)
                    if role_start<0: continue
                    role_mentions[mention]={"role_label":(B_label_id,I_label_id),"mentions":[(role_start,role_end)]}

            item_label_index=self.find_closest_roles(role_mentions)
            label_index.extend([(x[0],x[1],event_index) for x in item_label_index])
            start_indices=[x[0][0] for x in item_label_index]

            if concurrence_matrix!=None:
                word_length=concurrence_matrix.shape[0]
                for anchor in start_indices:
                    for start_index in start_indices:
                        if anchor not in offset_index_dict or start_index not in offset_index_dict:
                            continue
                        concurrence_matrix[offset_index_dict[anchor]][offset_index_dict[start_index]]=1

        # Sort by argument first index and event index.
        # label_index.sort(key=lambda x:(x[0][0],x[2]))
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


def share_argument_stat(data):
    with open("../data/fin_share_argument_stat.json","w") as jfile:
        for item in data:
            event_types=[event["event_type"] for event in item.get("event_list",[])]
            if len(event_types)!=len(set(event_types)):
                jfile.write(f"{json.dumps(item,ensure_ascii=False)}\n")

def mutli_events(data):
    with open("../data/multi_events.json","w") as jfile:
        for item in data:
            event_types=[event["event_type"] for event in item.get("event_list",[])]
            if 1!=len(event_types):
                jfile.write(f"{json.dumps(item,ensure_ascii=False)}\n")



if __name__ == '__main__':
    # schema=EventSchemaDict("../data/duee_fin_event_schema_sub.json")
    # print(schema.event_subject['股东减持'])
    # exit() 

    from collections import namedtuple

    config_class = namedtuple('config',['train_path','dev_path','test_path','pretrained_path','schema_path'])
    config = config_class("../data/multi_events.json","../data/duee_fin_dev.json","../data/duee_fin_test1.json","/storage/public/models/bert-base-chinese","../data/duee_fin_event_schema_sub.json")

    processor=NERProcessor(config)
    train_data=processor.get_train_data()
    with open("add_index_train.json","w") as f:
        for i in train_data:
            _,add_index_event_list=processor.event_schema.tokens_to_label_index(i["text"],i.get("event_list",[]),None,None)
            i["index_event_list"]=add_index_event_list
            f.write(json.dumps(i,ensure_ascii=False)+"\n")
    # mutli_events(train_data)
    # loader=processor.create_dataloader(train_data,batch_size=8)
    # print(miss_cnt)
    # for batch in loader:
    #     x = batch