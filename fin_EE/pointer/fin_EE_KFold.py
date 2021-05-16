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

from fin_EE_model import NERModel, NERPredictor
from configuration import Config
import utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    WORKING_DIR = "."
    DATA_DIR="/home/yadong/workspace/duie/fin_EE/data"

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=2, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=15, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--k", type=int, default=5, help="input batch size for training and test (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--crf_lr", type=float, default=0.1, help="crf learning rate (default: 0.1)")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")

    parser.add_argument("--use_bert", type=utils.str2bool, default=True,
                        help="whether to use bert training or not (default: True)")
    parser.add_argument("--use_crf", type=utils.str2bool, default=True,
                        help="whether to use crf layer training or not (default: True)")
    parser.add_argument("--split_long", type=utils.str2bool, default=True,
                        help="whether to split long text to multi")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default="{}/duee_fin_train.json".format(DATA_DIR),
                        help="train_path")
    parser.add_argument("--dev_path", type=str, default="{}/duee_fin_dev.json".format(DATA_DIR),
                        help="dev_path")
    parser.add_argument("--schema_path", type=str, default="{}/duee_fin_event_schema_sub.json".format(DATA_DIR),
                        help="schema_path")
    parser.add_argument("--test_path", type=str, default="{}/duee_fin_test1.json".format(DATA_DIR),
                        help="test_path")
    parser.add_argument("--ner_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ner_result_path")
    parser.add_argument("--ner_save_path", type=str,
                        default="{}/kweights".format(WORKING_DIR), help="ner_save_path")
    parser.add_argument("--pretrained_path", type=str,
                        default="/storage/public/models/chinese-roberta-wwm-ext-large".format(WORKING_DIR), help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="###", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="val_total_f1=1.513_pf1=0.748cf1=0.764_epoch=10_large.ckpt", help="ckpt name for test")
    args = parser.parse_args()
    

    # Init config
    config = Config.from_dict(args.__dict__)

    print('--------config----------')
    print(config)
    print('--------config----------')

    if config.is_train == True:
        # ============= train 训练模型==============
        for kno in range(config.k):
            print("start train model ...")
            model = NERModel(config,kno)

            # 设置保存模型的路径及参数
            ckpt_callback = ModelCheckpoint(
                dirpath=config.ner_save_path+f"/k{kno}",                           # 模型保存路径
                filename="{val_total_f1:.3f}_{pf1:.3f}{cf1:.3f}_{epoch}",   # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
                monitor='val_total_f1',                                      # 根据验证集上的准确率评估模型优劣
                mode='max',
                save_top_k=3,                                           # 保存得分最高的前两个模型
                verbose=True
            )

            # 设置训练器
            trainer = pl.Trainer(
                progress_bar_refresh_rate=1,
                max_epochs=config.max_epochs,
                callbacks=[ckpt_callback,utils.PrintLineCallback()],
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
        outfile_txt = os.path.join(config.ner_result_path, "k.json")

        # 开始测试，将结果保存至输出文件
        checkpoint_path = os.path.join(config.ner_save_path, config.test_ckpt_name)
        predictor = NERPredictor(checkpoint_path, config)
        # predictor.validation()
        predictor.generate_k_temp()
        predictor.generate_k_result(outfile_txt)
        print('\n', 'outfile_txt name:', outfile_txt)