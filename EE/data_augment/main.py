# coding=utf-8
import sys
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from configuration import Config
import utils
from module import ARModule

utils.set_random_seed(20210507)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    WORKING_DIR = "."
    DATA_DIR="/home/yadong/workspace/duie/EE/data"

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=3, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=20, help="the max epochs for training and test (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (default: 2e-5)")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="optimizer")

    # 下面参数基本默认
    parser.add_argument("--train_path", type=str, default="{}/duee_train.json".format(DATA_DIR),
                        help="train_path")
    parser.add_argument("--dev_path", type=str, default="{}/duee_dev.json".format(DATA_DIR),
                        help="dev_path")
    parser.add_argument("--test_path", type=str, default="{}/duee_test1.json".format(DATA_DIR),
                        help="test_path")
    parser.add_argument("--schema_path", type=str, default="{}/duee_event_schema.json".format(DATA_DIR),
                        help="schema_path")
    parser.add_argument("--ner_result_path", type=str, default="{}/result".format(WORKING_DIR),
                        help="ner_result_path")
    parser.add_argument("--ner_save_path", type=str,
                        default="{}/weights".format(WORKING_DIR), help="ner_save_path")
    parser.add_argument("--pretrained_path", type=str,
                        default="/storage/public/models/mt5-base-zh".format(WORKING_DIR), help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="###", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="val_f1=0.7680_epoch=13_global.ckpt", help="ckpt name for test")

    args = parser.parse_args()
    
    # Init config
    config = Config.from_dict(args.__dict__)

    print('--------config----------')
    print(config)
    print('--------config----------')

    if config.is_train == True:
        # ============= train 训练模型==============
        print("start train model ...")
        model = ARModule(config)
        early_stopping=EarlyStopping("loss",mode="min")

        # 设置保存模型的路径及参数
        ckpt_callback = ModelCheckpoint(
            dirpath=config.ner_save_path,                           # 模型保存路径
            filename="{loss:.4f}_{epoch}_global+large",   # 模型保存名称，参数ckpt_name后加入epoch信息以及验证集分数
            monitor='loss',                                      # 根据验证集上的准确率评估模型优劣
            mode='min',
            save_top_k=3,                                           # 保存得分最高的前两个模型
            verbose=True,
        )

        # 设置训练器
        trainer = pl.Trainer(
            progress_bar_refresh_rate=1,
            # resume_from_checkpoint = config.ner_save_path + '/val_f1=0.7519_epoch=10_global+sec+rope.ckpt',  # 加载已保存的模型继续训练
            max_epochs=config.max_epochs,
            callbacks=[ckpt_callback,early_stopping],
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
        outfile_txt = os.path.join(config.ner_result_path, config.test_ckpt_name[:-5] + ".json")

        # 开始测试，将结果保存至输出文件
        checkpoint_path = os.path.join(config.ner_save_path, config.test_ckpt_name)
        predictor = NERPredictor(checkpoint_path, config)
        predictor.generate_result(outfile_txt)
        print('\n', 'outfile_txt name:', outfile_txt)