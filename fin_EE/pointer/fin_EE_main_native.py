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

from fin_EE_model import NERModel, NERPredictor,PointerNet
from configuration import Config
from fin_EE_data_util import NERProcessor
import utils

utils.set_random_seed(20200819)
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    WORKING_DIR = "."
    DATA_DIR="/home/yadong/workspace/duie/fin_EE/data"

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=utils.str2bool, default=True, help="train the NER model or not (default: False)")
    parser.add_argument("--batch_size", type=int, default=4, help="input batch size for training and test (default: 8)")
    parser.add_argument("--max_epochs", type=int, default=40, help="the max epochs for training and test (default: 5)")
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
                        default="{}/weights".format(WORKING_DIR), help="ner_save_path")
    parser.add_argument("--pretrained_path", type=str,
                        default="/storage/public/models/chinese-roberta-wwm-ext".format(WORKING_DIR), help="pretrained_path")

    parser.add_argument("--ckpt_name",  type=str, default="###", help="ckpt save name")
    parser.add_argument("--test_ckpt_name",  type=str, default="val_f1=0.7395_epoch=10_split.ckpt", help="ckpt name for test")

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
        processor=NERProcessor(config)
        model = PointerNet(config,len(processor.event_schema.id2labels))
        model.cuda()
        origin_train_data = processor.get_train_data()[:3000]
        origin_dev_data=processor.get_dev_data()[:1000]
        train_data=processor.process_long_text(origin_train_data,512)
        dev_data=processor.process_long_text(origin_dev_data,512)
        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        train_loader = processor.create_dataloader(
            train_data, batch_size=config.batch_size, shuffle=True)
        valid_loader = processor.create_dataloader(
            dev_data, batch_size=config.batch_size, shuffle=False)


        criterion = torch.nn.BCELoss(reduction="sum")
        arg_list = [p for p in model.parameters() if p.requires_grad]
        print("Num parameters:", len(arg_list))
        optimizer=torch.optim.Adam(arg_list, lr=config.lr, eps=1e-8)

        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            for i in range(len(batch)-1):
                batch[i]=batch[i].cuda()
            input_ids, token_type_ids, attention_mask,offset_mapping, label_tensors,concurrence_tensors,label_text_index = batch
            optimizer.zero_grad()
            pointer,concurrence = model({"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attention_mask})
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            
            pointer_loss = criterion(pointer[attention_mask!=0,:], label_tensors[attention_mask!=0])
            concurrence_loss= criterion(concurrence[attention_mask!=0,:], concurrence_tensors[attention_mask!=0])
            loss=pointer_loss+concurrence_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            print(loss.item())
            # epoch_loss += loss.item()


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