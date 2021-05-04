#!/usr/bin/env python
import itertools
import random
import re
from typing import List
import argparse

import torch
import numpy as np
from datetime import datetime
import os

from pytorch_lightning.callbacks import Callback

class PrintLineCallback(Callback):


    def on_save_checkpoint(self, trainer, pl_module,checkpoint):
        print()
        return {}

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def findSubDir(base):
    '''
    返回base目录下一层子目录
    '''
    for root, ds, fs in os.walk(base):
        return ds

def check_dir_is_none(dir_path):
    files = []
    for file in findAllFile(dir_path):
        files.append(file)
    if len(files):
        return False
    else:
        return True


def is_alphabet(string):
    """判断一个字符串是否是英文字母"""
    for uchar in string:
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            continue
        else:
            return False
    return True

def change_date(time_str):
    '''
    转换时间格式，将字符串格式的时间或者整数类型的时间戳， 转换为标准时间。
    输入：'2020-03-11 19:13:22' 或 1605511400466
    输出：'2020-03-11T19:13:22.000+0800'
    '''
    if isinstance(time_str, int):   # 把时间戳转化为字符串时间: 1605511400466 -->  '2020-11-16 07:23:20'
        time_str = datetime.fromtimestamp(time_str / 1000.).strftime("%Y-%m-%d %H:%M:%S")
    time_str = time_str.replace(' ', 'T')
    time_str = time_str + '.000+0800'
    return time_str

def str2bool(v):
    '''
    将字符转化为bool类型
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_companies_from_ner_res(ner_res: List[tuple]) -> List[tuple]:
    """
    从 NER 识别结果中抽取公司。
    :param ner_res: 单条NER的模型的预测结果
            例如：[(w0, 'O'), (w1, 'B-COM'), (w2, 'I-COM'),.... (wn, 'O')],
    :return companies_and_indices: 句子中的所有公司实体。
            例如：[('百度公司', [0, 1, 2, 3]), ('阿里巴巴', [6, 7, 8, 9])]
    """
    companies, indices = [], []
    bucket_words, bucket_indices = [], []
    com_start = False
    for index, (word, label) in enumerate(ner_res):
        if label == 'B-COM':
            # Empty bucket.
            if com_start and bucket_words and len(bucket_words) > 1:  # 大于1，过滤掉单个字符的公司
                companies.append("".join(bucket_words))
                indices.append(bucket_indices)
            bucket_words, bucket_indices = [], []
            # Add to bucket.
            com_start = True
            bucket_words.append(word.replace('#_#', ''))
            bucket_indices.append(index)
        elif label == 'I-COM':
            if com_start:
                if is_alphabet(word) and is_alphabet(bucket_words[-1]):  # 前面是英文单词时，碰到新的单词添加空格
                    bucket_words.append(" ")
                bucket_words.append(word.replace('#_#', ''))  # 去除英文单词mask的"#"符号
                bucket_indices.append(index)
            else:
                bucket_words, bucket_indices = [], []
        elif label == 'O':
            # Empty bucket.
            if com_start and bucket_words and len(bucket_words) > 1:
                companies.append("".join(bucket_words))
                indices.append(bucket_indices)
            bucket_words, bucket_indices = [], []

            com_start = False

    if com_start and bucket_words:
        companies.append("".join(bucket_words))
        indices.append(bucket_indices)

    # 后处理，删除公司名中含有'某'的公司
    companies_and_indices = []
    not_com_list = ['支付宝']
    for com, idx in zip(companies, indices):
        if '某' in com:
            continue
        companies_and_indices.append((com, idx))

    return companies_and_indices


re_sentence_sp = re.compile('([﹒﹔；﹖﹗．。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def split_sentence(content, max_sen_length=512):
    '''
    将正文按句切分，返回句子列表
    '''
    s = content
    slist = []
    for i in re_sentence_sp.split(s):  # 将句子按照正则表达式切分
        if re_sentence_sp.match(i) and slist:  # 如果是标点符号，则添加到上一句末尾
            slist[-1] += i
        elif i.strip():  # 不是标点符号，也不是空字符串，则将句子添加到句子列表
            while len(i) >= max_sen_length - 2:  # 按句切分后句子长度大于BERT最大长度 (-2是因为句子开头和结尾要添加CLS和SEP)
                sub_i, i = i[:max_sen_length - 2], i[max_sen_length - 2:]
                slist.append(sub_i)
            slist.append(i)
    return slist


def batch_iter(iterable, batch_size):
    """
    Batch iter.
    """
    it = iter(iterable)
    batch = list(itertools.islice(it, batch_size))
    while batch:
        yield batch
        batch = list(itertools.islice(it, batch_size))


def write_row_in_sheet(row_index, row_dict, export_header, sheet, format=None):
    for i, head in enumerate(export_header):
        value = row_dict.get(head, "")
        sheet.write(row_index, i, value, format)
    return row_index + 1


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    s = "原标题：格力电器：拟以不低于30亿不超过60;；亿元回购公司股份 新京报讯 4月12日，珠海格力电器股份有限公司发布公告称，拟使用自有资金以集中竞价交易方式回购公司股份，回购股份的种类为公司发行的A股股份，资金总额不低于人民币30亿元（含）且不超过人民币60亿元（含）；回购股份价格不超过人民币70元/股（以下简称“本次回购”）。按本次回购资金最高人民币60亿元测算，预计可回购股份数量约为85714285股，约占公司目前总股本的1.42%；按本次回购资金最低人民币30亿元测算，预计可回购股份数量约为42857143股，约占公司目前总股本的0.71%。回购期限自公司董事会审议通过本次回购方案之日起不超过12个月，具体回购数量以回购期满时实际回购的股份数量为准，回购股份将用于员工持股计划或者股权激励。"
    print(extract_companies_from_ner_res([("xxx", "B-COM")]))
    print(split_sentence(""))
