# coding=utf-8
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# 添加src目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /working/financial_news_insight_system/src/NER
sys.path.append(os.path.dirname(BASE_DIR))              # 将src目录添加到环境

from conlleval import evaluate_conll_file
from EE_model import NERModel, NERPredictor
from configuration import Config
import utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    WORKING_DIR = "."

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=False, help="train the NER model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=8, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=20, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--crf_lr", type=float, default=0.1, help="crf learning rate (default: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")

    parser.add_argument("--use_bert", type=utils.str2bool, default=True,
                        help="whether to use bert training or not (default: True)")
    parser.add_argument("--use_crf", type=utils.str2bool, default=True,
                        help="whether to use crf layer training or not (default: True)")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default="{}/data/duee_train.json".format(WORKING_DIR),
                        help="train_path")
    parser.add_argument("--dev_path", type=str, default="{}/data/duee_dev.json".format(WORKING_DIR),
                        help="dev_path")
    parser.add_argument("--schema_path", type=str, default="{}/data/duee_event_schema.json".format(WORKING_DIR),
                        help="schema_path")
    parser.add_argument("--test_path", type=str, default="{}/data/duee_test1.json".format(WORKING_DIR),
                        help="test_path")
    parser.add_argument("--ner_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ner_result_path")
    parser.add_argument("--ner_save_path", type=str,
                        default="{}/weights".format(WORKING_DIR), help="ner_save_path")
    parser.add_argument("--pretrained_path", type=str,
                        default="/storage/public/models/chinese-roberta-wwm-ext".format(WORKING_DIR), help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="###", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="###_epoch=13_val_f1=70.7.ckpt", help="ckpt name for test")

    args = parser.parse_args()

    if args.ckpt_name == "###":         # 如果没有传入ckpt名字，根据设置自动构造ckpt模型保存名字
        ck_model_name = "BERT_" if args.use_bert else "LSTM_"   # 加入模型名信息
        ck_crf_name = "CRF_" if args.use_crf else "woCRF_"      # 加入crf信息

        ck_epochs_name = str(args.max_epochs)                   # epoch数目信息

    # Init config
    config = Config.from_dict(args.__dict__)
    if args.is_train:   # 训练模式下，将参数保存
        config.save_to_json_file(os.path.join(config.ner_save_path, "config.json"))

    print('--------config----------')
    print(config)
    print('--------config----------')

    if config.is_train == True:
        # ============= train 训练模型==============
        print("start train model ...")
        model = NERModel(config)

        # 设置保存模型的路径及参数
        ckpt_callback = ModelCheckpoint(
            dirpath=config.ner_save_path,                           # 模型保存路径
            filename=config.ckpt_name + "_{epoch}_{val_f1:.1f}",   # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
            monitor='val_f1',                                      # 根据验证集上的准确率评估模型优劣
            mode='max',
            save_top_k=2,                                           # 保存得分最高的前两个模型
            verbose=True,
        )

        # 设置训练器
        trainer = pl.Trainer(
            progress_bar_refresh_rate=1,
            resume_from_checkpoint = config.ner_save_path + '/###_epoch=1_val_f1=69.4.ckpt',  # 加载已保存的模型继续训练
            max_epochs=config.max_epochs,
            callbacks=[ckpt_callback],
            checkpoint_callback=True,
            gpus=1,
            distributed_backend='dp',
            profiler=True,
        )

        # 开始训练模型
        trainer.fit(model)

        # 只训练CRF的时候，保存最后的模型
        # if config.use_crf and config.first_train_crf == 1:
        #     trainer.save_checkpoint(os.path.join(config.ner_save_path, 'crf_%d.ckpt' % (config.max_epochs)))
    else:
        # ============= test 测试模型==============
        print("\n\nstart test model...")

        '''
        设置输出文件名 得到结果为 
            1).txt 测试数据的字符输出文件 
            共三列，分别是：字符token， 真实BIO标签， 模型预测BIO标签
                eg:
                    浙 B-COM B-COM
                    江 I-COM I-COM
                    龙 I-COM I-COM
                    盛 I-COM I-COM
                    于 O O
                    2020 O O
                    年 O O
                    8 O O
                    月 O O
                    18 O O
                    日 O O
                    
            2).tsv 根据txt内容，整理得到句子+公司名文件，与训练输入文件保持一致
            共两列，以制表符分割，分别是 句子 公司名列表
                eg:
                浙江龙盛于2020年8月18日披露中报，公司2020上半年实现营业总收入75.9亿	['浙江龙盛']
        '''
        outfile_txt = os.path.join(config.ner_result_path, config.test_ckpt_name[:-5] + ".json")

        # 开始测试，将结果保存至输出文件
        checkpoint_path = os.path.join(config.ner_save_path, config.test_ckpt_name)
        predictor = NERPredictor(checkpoint_path, config)
        predictor.generate_result(outfile_txt)
        print('\n', 'outfile_txt name:', outfile_txt)