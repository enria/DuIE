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

class SubjectDataset(torch.utils.data.Dataset):
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


class SubjectProcessor(DataProcessor):
    """
       从NER数据文件/data/NER_data下读取数据，生成NER训练数据dataloader，返回给模型
    """

    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.test_path=config.test_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.subject_schema=SubjectSchemaDict(config.schema_path)

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
        label_tensor=torch.zeros(len(offset),len(self.subject_schema.id2labels))

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
    

    def create_dataset(self, data, batch_size, shuffle=False, max_length=512,is_test=False):
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
        label_text_index=[]
        for index,offset in tqdm(enumerate(inputs['offset_mapping']),total=len(data)):
            if is_test:
                event_list=[]
                label_index=[]
                label_tensors.append(torch.zeros(1))
                label_text_index.append([(x[0],x[1]) for x in label_index])
            else:
                offset_index_dict={} # text index to offset index
                for i,o in enumerate(offset):
                    if o[-1]==0:
                        continue
                    for oi in range(o[0],o[-1]):
                        offset_index_dict[int(oi)]=i
                event_list=data[index].get("spo_list",[])
                concurrence_matrix=torch.zeros((offset.shape[0],offset.shape[0]))
                label_index=self.subject_schema.tokens_to_label_index(text[index],event_list,offset_index_dict,concurrence_matrix)
                label_tensor=self.from_offset_to_label_tensor(offset,label_index,offset_index_dict)
                label_tensors.append(label_tensor)
                label_text_index.append([(x[0],x[1]) for x in label_index])

        dataset=SubjectDataset(inputs["input_ids"],
                        inputs["token_type_ids"],
                        inputs["attention_mask"],
                        inputs["offset_mapping"],
                        label_tensors,
                        label_text_index)

        return dataset
    
    def from_label_tensor_to_label_index(self,label_tensor_list,offset_mapping_list,threshold=0.2):
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

            result.append(item_result)

        return result

class SubjectSchemaDict(object):
    def __init__(self,schema_path):
        self.schema_path=schema_path
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
                event_type=event_schema["predicate"]
                role_list.append(f"{event_type}:subject")

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
    
    def tokens_to_label_index(self,text:str,label,offset_index_dict,concurrence_matrix):
        label_index=[]
        for event_index,event in enumerate(label):
            event_type=event["predicate"]
            subject_mention=event["subject"].strip()
            if not subject_mention:
                continue
            B_label_id=self.label2ids[f"B-{event_type}:subject"]
            I_label_id=self.label2ids[f"E-{event_type}:subject"]

            role_start=text.find(subject_mention)
            role_end=role_start+len(subject_mention)
            if role_start<0: continue

            item_label_index=[((role_start,role_end),(B_label_id,I_label_id))]

            label_index.extend([(x[0],x[1],event_index) for x in item_label_index])

        return label_index
        

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self,input_ids,token_type_ids,attention_mask,offset_mapping,label_tensor,subject_labe_texts,line_no):
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.offset_mapping=offset_mapping
        self.label_tensor=label_tensor
        self.subject_labe_texts=subject_labe_texts
        self.line_no=line_no

    def __getitem__(self, index):
        return (self.input_ids[index],
               self.token_type_ids[index],
               self.attention_mask[index],
               self.offset_mapping[index],
               self.label_tensor[index],
               self.subject_labe_texts[index],
               self.line_no[index])

    def __len__(self):
        return self.input_ids.size(0)

    @staticmethod
    def collate_fn(batch):
        batch = list(zip(*batch))
        for i in range(len(batch)-2):
            batch[i]=torch.stack(batch[i])
        return batch


class ObjectProcessor(DataProcessor):
    """
       从NER数据文件/data/NER_data下读取数据，生成NER训练数据dataloader，返回给模型
    """

    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path=config.dev_path
        self.test_path=config.test_path
        self.tokenizer = tokenizer
        self.object_schema=ObjectSchemaDict(config.schema_path)

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
    
    def from_offset_to_label_index_list(self,offset,label_index):
        offset_index=1
        label_index_list=torch.zeros(len(offset),dtype=torch.long)
        for role_index in label_index:
            if role_index[0][0]<offset[offset_index][0]:
                continue
            while offset_index<len(offset):
                if offset[offset_index][0]>=role_index[0][0]:
                    label_index_list[offset_index]=role_index[1][0]
                    break
                offset_index+=1

            offset_index+=1
            while offset_index<len(offset):
                if offset[offset_index][1]<=role_index[0][1]:
                    label_index_list[offset_index]=role_index[1][1]
                    offset_index+=1
                else:
                    break
            
            if offset_index>=len(offset):
                break
        else:
            pass #  TODO 统计覆盖
        return label_index_list
    

    def create_dataset(self, data, max_length=512):

        subject_labe_texts=[]
        label_index_list=[]
        line_no=[]
        for no,item in tqdm(enumerate(data),total=len(data),desc="Add Subject Label"):
            spo_list=item.get("spo_list",[])
            tls=self.object_schema.tokens_to_label_index(item["text"],spo_list)
            subject_labe_texts.extend([t[0] for t in tls])
            label_index_list.extend([t[1] for t in tls])
            line_no.extend([no]*len(tls))

        tokenizer = self.tokenizer
        max_length = min(max_length, max([len(s) for s in subject_labe_texts]))
        print("max sentence length: ", max_length)

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            subject_labe_texts,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        label_tensors = []
        for offset,label_index in tqdm(zip(inputs['offset_mapping'],label_index_list),total=len(subject_labe_texts),desc="Convert Label to Tensor"):
            label_tensor=self.from_offset_to_label_index_list(offset,label_index)
            label_tensors.append(label_tensor)

        dataset=ObjectDataset(inputs["input_ids"],
                        inputs["token_type_ids"],
                        inputs["attention_mask"],
                        inputs["offset_mapping"],
                        label_tensors,
                        subject_labe_texts,
                        line_no)

        return dataset
    
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

            result.append(item_result)

        return result

class ObjectSchemaDict(object):
    def __init__(self,schema_path):
        self.schema_path=schema_path
        self.label2ids={}
        self.id2labels={}

        self.predicates=[]
        self.subject_tokens=[]
        self.complicated_spo=set()

        self.load_schema()

    def load_schema(self):
        role_list=[]
        with open(self.schema_path) as schema:
            for line in schema:
                if not line:
                    continue
                spo_schema=json.loads(line)
                predicate=spo_schema["predicate"]
                self.predicates.append(predicate)
                for key, desc in spo_schema["object_type"].items():
                    role_list.append(f"{predicate}:{key}")
                if len(spo_schema["object_type"])>1:
                    self.complicated_spo.add(predicate)

        self.label2ids["O"]=0
        self.id2labels[0]="O"
        for index,role in enumerate(role_list):
            self.label2ids["B-"+role]=index*2+1
            self.label2ids["I-"+role]=index*2+2
            self.id2labels[index*2+1]="B-"+role
            self.id2labels[index*2+2]="I-"+role
        
        for predicate in self.predicates:
            self.subject_tokens.append(self.predicate_head_token(predicate))
            self.subject_tokens.append(self.predicate_tail_token(predicate))
    
    def find_all_sub(self,a_str,sub,end=510):
        start = 0
        while True:
            start = a_str.find(sub, start,end)
            if start == -1: return []
            yield (start,start+len(sub))
            start += len(sub)
    
    def predicate_head_token(self,predicate):
        return f"<s-{self.predicates.index(predicate)}>"
    
    def predicate_tail_token(self,predicate):
        return f"</s-{self.predicates.index(predicate)}>"
    
    
    def tokens_to_label_index(self,text:str,label):
        subject_dict={}

        for spo_index,spo in enumerate(label):
            spo_type=spo["predicate"]
            subject_mention=spo["subject"].strip()
            dict_key=f"{spo_type}-{subject_mention}"
            if dict_key not in subject_dict:
                subject_label_text=text.replace(subject_mention,
                                   f"{self.predicate_head_token(spo_type)}{subject_mention}{self.predicate_tail_token(spo_type)}")
                subject_dict[dict_key]=(subject_label_text,[])
            
            
            subject_label_text,object_labels=subject_dict[dict_key]

            for key,value in spo.get("object",{}).items():
                mention=value.strip()
                role_start=subject_label_text.find(mention)
                role_end=role_start+len(mention)
                B_label_id=self.label2ids["B-"+spo_type+":"+key]
                I_label_id=self.label2ids["I-"+spo_type+":"+key]
                object_labels.append(((role_start,role_end),(B_label_id,I_label_id)))

        if not subject_dict:
            return [[text,[]]]
        
        return subject_dict.values()


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