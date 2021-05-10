# -*- coding: utf-8 -*-
# @File    :   data_utils.py
# @Time    :   2021/03/16 22:03:10
# @Author  :   Qing 
# @Email   :   sqzhao@stu.ecnu.edu.cn

import os
import sys
import json
import random
import numpy as np 

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import pytorch_lightning as pl

from transformers import BertTokenizerFast, AutoTokenizer, MT5TokenizerFast
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = CUR_DIR[:CUR_DIR.index('src')]


class BaseDataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = raw_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("This is base class! Implement your own Dataset class!")

    def load_fn(self, batch):
        raise NotImplementedError


def load_data(path, shuffle=False):
    """
    {
        "data":[{
            "title":"",
            "paragraphs":[
                {
                    "qas": [
                        {
                            "type": "in-domain", 
                            "question": "怀孕初期不小心吃了扁桃仁", 
                            "id": "e0285179a076cfb1e667042de441d1b5", 
                            "answers": [
                                {
                                    "text": "孕妇吃扁桃仁一般情况下是不会导致胎儿发生病变的,通常如果不食用过量的话对宝宝是没有什么影响的,适量使用也是可以给宝宝带来很好的效果", 
                                    "answer_start": 0
                                }
                            ], 
                            "is_impossible": false
                        }
                    ], 
                    "context": "孕妇吃扁桃仁一般情况下是不会导致胎儿发生病变的,通常如果不食用过量的话对宝宝是没有什么影响的,适量使用也是可以给宝宝带来很好的效果,不过孕妇一定要注意膳食搭配,营养均衡这样才能生出一个健康的宝宝。", 
                    "title": "孕妇吃扁桃仁胎儿畸形是为什么_妈妈网小百科"
                },
                ...
            ]
        }]
    }
    """
    R = []
    with open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
        for entry in obj['data']:
            for sample in entry['paragraphs']:
                context = sample['context'].strip()
                title = sample.get("title", "")
                for qas in sample['qas']:
                    R.append([
                        title,
                        context, 
                        qas['id'], 
                        qas.get('type', ""),
                        qas['question'].strip(),
                        [ a['text'].strip() for a in qas.get('answers', [])],
                        [ a['answer_start'] for a in qas.get('answers', [])],
                        qas.get('is_impossible', False)
                    ])
    print(f"{path} loaded with {len(R)} samples!")
    if shuffle:  # shuffle only for train data
        random.shuffle(R)
    return R

class MRC(BaseDataset):
    def __getitem__(self, index):
        sample = self.data[index]
        title, context, qid, qtype, question, answers, answer_starts, is_impossible = sample
        if answers == []:                  # 测试时充填答案部分的标注
            return qid, context, question, "Testing Answer", -1, -1, is_impossible, sample
        answer = answers[0]
        if is_impossible:
            start_char, end_char = -1, -1
        else:
            start_char, end_char = answer_starts[0], answer_starts[0] + len(answer)
        return qid, context, question, answer, start_char, end_char, is_impossible, sample

    def get_start_end_index(self, start_chars, end_chars, input_ids, offset_mapping, is_impossible):
        s_idx, e_idx, answerable = [], [], []
        for s, e, inputids, mapping, ip in zip(start_chars, end_chars, input_ids, offset_mapping, is_impossible):
            # 不可回答
            if s == -1 == e and ip:
                s_idx.append(0); e_idx.append(0); 
                answerable.append(0)
                continue
            sep_idx = inputids.index(self.tokenizer.sep_token_id)
            sent_mapping = mapping[sep_idx:]  # 从第一个SEP Token后开始找 

            si, ei = None, None
            for i, (_s, _e) in  enumerate(sent_mapping):
                if _s == s and si is None:
                    si = i + sep_idx
                if _e == e and ei is None:
                    ei = i + sep_idx
            
            # 开始位置找到了，但是结束位置被截断的情况，设为不可回答
            if si is None or ei is None:
                s_idx.append(0); e_idx.append(0)
                answerable.append(0)
            else:
                s_idx.append(si); e_idx.append(ei)
                answerable.append(1)
        return s_idx, e_idx, answerable

    def load_fn(self, batch):
        qid, context, question, answer, start_chars, end_chars, is_impossible, sample = zip(*batch)
        tokenized_dict = self.tokenizer(question, context, max_length=512, padding=True, truncation=True, return_offsets_mapping=True)
        R = {
            'qids': qid,
            'input_ids': torch.tensor(tokenized_dict.input_ids),
            'token_type_ids': torch.tensor(tokenized_dict.token_type_ids),
            'attention_mask': torch.tensor(tokenized_dict.attention_mask),
            'offset_mapping': tokenized_dict.offset_mapping,
            'context': context,
            'gold': answer
        }
        if answer[0] == "Testing Answer":  # Testing Mode for prediction
            return R

        s_idx, e_idx, answerable = self.get_start_end_index(start_chars, end_chars, tokenized_dict.input_ids, tokenized_dict.offset_mapping, is_impossible) 
        assert len(s_idx) == len(e_idx) == len(batch), f"{s_idx}||{e_idx}"
        # 增加训练时需要的一些数据域
        R.update({
            'sample':sample,
            's': torch.tensor(s_idx).long(),
            'e': torch.tensor(e_idx).long(),
            'answerable': torch.tensor(answerable).long(),
        })
        return R

class TitleAheadMRC(MRC):
    """ 据实验发现将title拼接在context前会对效果有提升 """
    def __getitem__(self, index):
        title = self.data[index][0]
        qid, context, question, answer, start_char, end_char, is_impossible, sample = super().__getitem__(index)
        title_prefix = title
        title_len = len(title_prefix)
        return qid, title_prefix+context, question, answer, start_char+title_len, end_char+title_len, is_impossible, sample


class EOSTokenMRC(MRC):
    """ 在context最后加一个`[EOS]`token，使用它来做无答案情况下s,e的指向位置，而不是基线中指向[CLS] """
    def __getitem__(self, index):
        qid, context, question, answer, start_char, end_char, is_impossible, sample = super().__getitem__(index)
        return qid, context+"[EOS]", question, answer, start_char, end_char, is_impossible, sample

    def load_fn(self, batch): # TODO:
        return super().load_fn(batch)

class HomoDict(object):
    _instance = None
    default_data_path = os.path.join(PROJ_DIR, "resource/chinese_homophone_char.txt")
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)   
        else:
            print(f"pronunce num:{HomoDict._instance.pronunce_num}, word num:{HomoDict._instance.word_num}. Singleton created!")
        return cls._instance
    
    def __init__(self, homodata_path=None):
        self.word2pronunce_id = {
            '[PAD]' : 0,
            '[CLS]' : 1,
            '[SEP]' : 2,
            '[UNK]' : 3,
            '通配读音': 4
        }
        self.id2pronunce = {v: k for k,v in self.word2pronunce_id.items()}
        
        # 使用默认值
        homodata_path = HomoDict.default_data_path if homodata_path is None else homodata_path
        self.read_homophone_data(homodata_path)
        self.pronunce_num = len(self.id2pronunce)
        self.word_num = len(self.word2pronunce_id)

    def read_homophone_data(self, homedata_path):
        """ 0 为PAD保留 """
        with open(homedata_path, 'r', encoding='utf8') as f:
            for line in f:
                k, *v = line.strip().split()

                new_id = len(self.id2pronunce)
                self.id2pronunce[new_id] = k
                for word in v:
                    self.word2pronunce_id[word] = new_id

        
    def pronunce_id(self, word):
        """ 如果该token不在词表中，返回'通配读音'对应的id """
        word = '通配读音' if word not in self.word2pronunce_id else word
        return self.word2pronunce_id[word]

    def __repr__(self):
        return f"{self.__dict__}"
            

class HomophoneMRC(MRC):
    def __init__(self, data, tokenizer):
        super().__init__(data, tokenizer)
        self.homophone = HomoDict()

    def __getitem__(self, index):
        sample = self.data[index]
        title, context, qid, qtype, question, answers, answer_starts, is_impossible = sample
        if answers == []:                  # 测试时充填答案部分的标注
            return qid, context, question, "Testing Answer", -1, -1, is_impossible, sample
        answer = answers[0]
        if is_impossible:
            start_char, end_char = -1, -1
        else:
            start_char, end_char = answer_starts[0], answer_starts[0] + len(answer)
        return qid, context, question, answer, start_char, end_char, is_impossible, sample

    def convert_ids_to_pronunce_ids(self, x):
        return [self.homophone.pronunce_id(tok) for tok in self.tokenizer.convert_ids_to_tokens(x)]

    def load_fn(self, batch):
        super_batch =  super().load_fn(batch)
        super_batch["pronunce_ids"] = torch.tensor(list(map(self.convert_ids_to_pronunce_ids, super_batch['input_ids'])))
        return super_batch


class PosTagMRC(MRC):
    """ 增加pos-tag的Embedding层，需要为每个input_id对应一个额外的pos-tag_id """
    def __init__(self, raw_data, tokenizer):
        super().__init__(raw_data, tokenizer)
    
    def convert_ids_to_postag_ids(self, x):
        pass


class T5YesNo(BaseDataset):
    """ 用T5模型直接生成Yes, No判断是否可回答，大概率收敛向一个极点，即全部Yes或全部No """
    def __getitem__(self, index):
        sample = self.data[index]
        title, context, qid, qtype, question, answers, answer_starts, is_impossible = sample
        if answers == []:
            return qid, context, question, "Testing Answer", is_impossible, sample
        answer = "no" if  is_impossible else "yes"
        return qid, context, question, answer, is_impossible, sample

    def load_fn(self, batch):
        qid, context, question, answer, is_impossible, sample = zip(*batch)
        tokenized_input = self.tokenizer(list(question), list(context), padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokenized_target = self.tokenizer(list(answer), padding=True, truncation=True, max_length=5, return_tensors='pt')
        R = {
            'qids': qid,
            'input_ids': tokenized_input.input_ids,
            'attention_mask': tokenized_input.attention_mask,
            'target_ids' : tokenized_target.input_ids,
            'sample': sample
        }
        return R


# from torch_geometric.data import Dataset


class MRCData(pl.LightningDataModule):
    datasetmap = {
        'mrc':      (MRC, BertTokenizerFast),
        'squad':    (MRC, BertTokenizerFast),
        'mysquad':  (MRC, BertTokenizerFast),
        'title':    (TitleAheadMRC, BertTokenizerFast),
        'homoe':    (HomophoneMRC, BertTokenizerFast),
        't5yesno':  (T5YesNo, MT5TokenizerFast),
        
    }
    def __init__(self, config) -> None:
        super().__init__()
        self.data_dir = config.data_dir
        self.config = config
        # self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
        self.tokenizer = self.datasetmap[config.dataset][1].from_pretrained(config.pretrained)
        self.ds_class = self.datasetmap[config.dataset][0]

        self.create_data(config.mode)

    def create_data(self, mode=None):
        
        if mode in ['train', 'resume', 'cv'] or mode is None:
            self.train = load_data(self.config.train_path, shuffle=True)
            self.val =  load_data(self.config.val_path)

            # 从增强的数据中抽取10%，8：2分到train和val中，必要时可手动修改
            if self.config.fault_tolerant is not None and self.config.cklt_data:
                for path in self.config.fault_tolerant:
                    data = load_data(path)
                    n = len(data)
                    random.shuffle(data)
                    data = random.sample(data, int(n*0.1))
                    self.train += data[:int(n*0.8)]
                    self.val += data[int(n*0.8):]

        if mode is None or mode in ['test', 'eval', 'ensemble', 'ptv']:
            self.test =  load_data(self.config.test_path)
    
    
    def set_dataset(self, dataset=None):
        """ call before setup() """
        if dataset is not None:
            self.ds_class = dataset
        else:
            self.ds_class = self.datasetmap[self.config.dataset][0]

    @property
    def all_samples(self):
        """ 返回训练阶段用到的所有的数据 (用于KFold等操作)"""
        return self.train + self.val

    def rearrange_by_indices(self, train_idx, val_idx):
        """ 通过索引重新划分self.train 和 self.val """
        self.train = [ sample for i, sample in enumerate(self.all_samples) if i in train_idx]
        self.val = [ sample for i, sample in enumerate(self.all_samples) if i in val_idx]


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.ds_class(self.train, self.tokenizer)
            self.valset = self.ds_class(self.val, self.tokenizer)
            
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.ds_class(self.test, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.train_bsz,
            shuffle=True,
            collate_fn=self.trainset.load_fn,
            num_workers=4,
            # sampler=self.trainset.get_sampler()
        )

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.config.val_bsz, 
            shuffle=False,
            collate_fn=self.valset.load_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config.test_bsz, 
            shuffle=False,
            collate_fn=self.testset.load_fn, num_workers=4)


example_config_dict =  {
    "mode": 'train',
    "pretrained": "/pretrains/pt/hfl-chinese-roberta-wwm-ext",
    "dataset": "title",
    "train_path": '/home/qing/Competitions/BaiduMRC2021/data/train.json',
    # "train_path": '/home/qing/Competitions/BaiduMRC2021/data/merge/train.json',
    "val_path": '/home/qing/Competitions/BaiduMRC2021/data/dev.json',
    "test_path":'/home/qing/Competitions/BaiduMRC2021/data/official/test1.json',
    "fault_tolerant": None,
    "train_bsz": 4,
    "val_bsz": 4,
    "test_bsz": 4,
    "data_dir": "./data"
}

if __name__ == '__main__':
    
    from utils import Config

    config = Config.from_dict(example_config_dict)
    dm = MRCData(config)
    dm.setup('fit')
    # answerable = []
    # for x in dm.valset.data:
    #     print(x)
    #     break

    # print(len(dm.val_dataloader()))
    for batch in dm.val_dataloader():
        # print(batch)
        # break
        print(batch['s'].size(), batch['answerable'].size())
        for s, e, ss, ids, cc,m  in zip(batch['s'].numpy().tolist(), batch['e'].numpy().tolist(), batch['sample'], batch['input_ids'], batch['context'], batch['offset_mapping']):
            s_char = m[s][0]
            e_char = m[e][-1]
            print(ss[-3], "".join(dm.tokenizer.convert_ids_to_tokens(ids[s:e+1])), ss[-1], f"{s}--{e}", cc[s_char:e_char])
        # break
        # answerable.extend(batch['answerable'].numpy().tolist())
        
    #     for s, e in zip(batch['s'], batch['e']):
    #         if s == e == 0:
    #             answerable.append(0)
    #         else:
    #             answerable.append(1)
    # print(sum(answerable)/ len(answerable), len(answerable))
    # print(len([_ for _ in answerable if _ == 0]))
