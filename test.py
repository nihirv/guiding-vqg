import argparse
from pytorch_lightning import Trainer
from data_loader import get_loader
import os
import torch
from main import TrainVQG
from transformers.models.bert.tokenization_bert import BertTokenizer
from tokenizers.processors import TemplateProcessing

import numpy as np
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_rootdir", type=str, default="lightning_logs",
                        help="Root location of model checkpoints")
    parser.add_argument("--model_v_num", type=int, default=0,
                        help="model version number to load in")
    parser.add_argument("--val_dataset", type=str,
                        default="/data/nv419/VQG_DATA/processed/iq_val_dataset.hdf5")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_file", type=str,
                        default="model_outputs.json")
    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    args.device = device
    model_path_dir = os.path.join(args.model_rootdir, "version_{}".format(args.model_v_num), "checkpoints")
    model_ckpt = None
    for file in os.listdir(model_path_dir):
        if file.endswith(".ckpt"):
            model_ckpt = file  # returns the latest checkpoint in case of multiple files
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],)
    test_data_loader = get_loader(os.path.join(
        os.getcwd(), args.val_dataset), tokenizer, args.batch_size, shuffle=True, num_workers=8)
    model = TrainVQG.load_from_checkpoint(os.path.join(model_path_dir, model_ckpt)).to(device)
    trainer = Trainer(gpus=1)
    # image_ids1 = model.final_output["image_ids"]
    # gts1 = model.final_output["gts"]
    # preds1 = model.final_output["preds"]
    # print(len(image_ids1))
    # print(len(preds1))
    # print(len(gts1))
    json.dump(model.outputs_to_save, open(args.output_file, 'w'))
    trainer.test(model, test_dataloaders=test_data_loader)

    json.dump(model.outputs_to_save, open(args.output_file, 'w'))

    for k, scores in model.test_scores.items():
        print(k, np.mean(scores))

    # image_ids1 = model.final_output["image_ids"]
    # gts1 = model.final_output["gts"]
    # preds1 = model.final_output["preds"]
    # print(len(image_ids1))
    # print(len(preds1))
    # print(len(gts1))
    # with open('output2.txt', 'w') as out:
    #     for i, imgid in enumerate(image_ids1):
    #         sss = str(imgid) + "\t" + gts1[i] + "\t" + preds1[i] + "\n"
    #         out.write(sss)