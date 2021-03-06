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

surpass_role_cnt=0
role_cnt=0
overlap_role_cnt=0
role_start_indices=[]


class NERProcessor(DataProcessor):
    """
        从NER数据文件/data/NER_data下读取数据，生成NER训练数据dataloader，返回给模型
    """

    def __init__(self, config, tokenizer=None):
        self.train_path = config.train_path
        self.dev_path = config.dev_path
        self.test_path=config.test_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)
        self.event_schema=EventSchemaDict(config.schema_path)
    
    def read_json_data(self,filename,add_title=False):
        data = []
        with open(filename, 'r') as fjson:
            for line in fjson:
                item = json.loads(line)
                data.append(item)
        return data
    
    def process_long_text(self,data,split_max_length):
        if split_max_length<=512:
            return data
        
        new_dataset=[]

        for item in data:
            if len(item["text"])<=512:
                new_dataset.append(item)
            else:
                pre_item=dict(item)
                pre_item["text"]=item["text"][:512]
                new_dataset.append(pre_item)
                post_item=dict(item)
                last_index=min(split_max_length,len(item["text"]))
                title=f"原标题：{item['title']}\n"
                post_item["text"]=title+item["text"][last_index-(512-len(title)):last_index]
                new_dataset.append(post_item)
        
        return new_dataset

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
        label_index_list=[0]*len(offset)
        for role_index in label_index:
            if role_index[0][0]<offset[offset_index][0]:
                global overlap_role_cnt
                overlap_role_cnt+=1
                continue
            while offset_index<len(offset):
                if offset[offset_index][1]>role_index[0][0]:
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

        return label_index_list
    
    def create_dataloader(self, data, batch_size, shuffle=False, max_length=512):
        tokenizer = self.tokenizer

        # 2. 分别对句子和公司进行编码表示
        text = [d["text"] for d in data]
        max_length = min(max_length, max([len(s) for s in text]))

        inputs = tokenizer(     # 得到文本的编码表示（句子前后会加入<cls>和<sep>特殊字符，并且将句子统一补充到最大句子长度
            text,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True
        )

        # 3. 在句子中查找公司，得到句子的标注BIO编码 labels：将公司名标注为B和I, 非公司名字段标注为O
        labels = []
        for index,offset in enumerate(inputs['offset_mapping']):
            event_list=data[index].get("event_list",[])
            label_index=self.event_schema.tokens_to_label_index(data[index]["text"],event_list)
            label_index_list=self.from_offset_to_label_index_list(offset,label_index)
            labels.append(label_index_list)

        # 4. 将得到的句子编码和BIO转为dataloader，供模型使用
        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(inputs["input_ids"]),          # 句子字符id
            torch.LongTensor(inputs["token_type_ids"]),     # 区分两句话，此任务中全为0，表示输入只有一句话
            torch.LongTensor(inputs["attention_mask"]),     # 区分是否是pad值。句子内容为1，pad为0
            torch.LongTensor(inputs["offset_mapping"]),     # 区分是否是pad值。句子内容为1，pad为0
            torch.LongTensor(labels),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=4,
        )
        return dataloader

class EventSchemaDict(object):
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
                event_type=event_schema["event_type"]

                #添加事件触发词
                role_list.append(event_type+":"+"TRG") 

                for role in event_schema["role_list"]:
                    role_list.append(event_type+":"+role["role"])

        self.label2ids["O"]=0
        self.id2labels[0]="O"
        for index,role in enumerate(role_list):
            self.label2ids["B-"+role]=index*2+1
            self.label2ids["I-"+role]=index*2+2
            self.id2labels[index*2+1]="B-"+role
            self.id2labels[index*2+2]="I-"+role
    
    def tokens_to_label_index(self,text:str,label):
        label_index=[]
        for event_index,event in enumerate(label):
            event_type=event["event_type"]
            for role in event["arguments"]+[{"role":"TRG","argument":event["trigger"]}]:# 将trigger也作为角色
                role_start=text.find(role["argument"])
                if role_start>512:
                    global surpass_role_cnt
                    surpass_role_cnt+=1
                if role_start<0:
                    continue
                global role_start_indices
                role_start_indices.append(role_start)
                role_end=role_start+len(role["argument"])
                B_label_id=self.label2ids["B-"+event_type+":"+role["role"]]
                I_label_id=self.label2ids["I-"+event_type+":"+role["role"]]
                label_index.append(((role_start,role_end),(B_label_id,I_label_id),event_index))
        label_index.sort(key=lambda x:(x[0][0],x[2]))
        return label_index
    

def text_length_hist(train_data):
    text_length=[len(item["text"]) for item in train_data]
    exceed=list(filter(lambda x: x>=512,text_length))
    print(len(exceed)/len(text_length))

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib

    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    data = np.random.randn(10000)
    plt.hist(text_length, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("区间")
    plt.ylabel("频数/频率")
    plt.title("频数/频率分布直方图")
    plt.savefig("hist.png")


if __name__ == '__main__':

    from collections import namedtuple

    config_class = namedtuple('config',['train_path','dev_path','test_path','pretrained_path','schema_path'])
    config = config_class("../data/duee_fin_train.json","../data/duee_fin_dev.json","../data/duee_fin_test1.json","/storage/public/models/bert-base-chinese","../data/duee_fin_event_schema.json")

    processor=NERProcessor(config)
    train_data=processor.get_train_data()
    # loader=processor.create_dataloader(train_data,8)
    text_length_hist(train_data)
    # print(overlap_role_cnt,len(role_start_indices))
    # text_length_hist(role_start_indices)
    # print(surpass_role_cnt,role_cnt)

    
