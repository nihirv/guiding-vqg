import argparse
import os

import numpy as np
from tokenizers.processors import TemplateProcessing
from data_loader import get_loader
from typing import Any, List
from copy import deepcopy
import torch
from model import VQGModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers.models.bert.tokenization_bert import BertTokenizer
from nlg_eval.nlgeval import NLGEval
from TextGenerationEvaluationMetrics.multiset_distances import MultisetDistances
from operator import itemgetter
import math

torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


class TrainVQG(pl.LightningModule):
    def __init__(self, args, tokenizer: BertTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.tokenizer = tokenizer
        self.latent_transformer = False
        self.model = VQGModel(args, self.tokenizer)

        self.iter = 0
        self.kliter = 0
        self.test_scores = {}
        self.nlge = NLGEval(no_glove=True, no_skipthoughts=True)

        self.val_losses = {"total loss": [], "rec loss": [], "kl loss": []}
        self.bleus = []
        self.msjs = []
        self.fbds = []

    def forward(self, batch, val=False):

        images = batch["images"]
        question_ids = batch["question_ids"]
        question_attention_masks = batch["question_attention_masks"]
        input_ids = batch["qa_inference_ids"]
        input_attention_masks = batch["qa_inference_attention_masks"]
        object_features = batch["object_features"]
        object_locations = batch["object_locations"]

        if self.args.variant in ("ifD-ifD"):  # RETRAIN THIS
            input_ids = batch["caption_ids"]
            input_attention_masks = batch["caption_attention_masks"]

        if self.args.variant in ("icf-icf"):
            input_ids = batch["category_only_ids"]
            input_attention_masks = batch["category_only_attn_masks"]

        loss = self.model(images, question_ids, question_attention_masks, input_ids, input_attention_masks, object_features, object_locations)
        return loss

    def calculate_losses(self, loss, kld, r=0.5):
        loss_rec = loss
        if kld is None:
            total_loss = loss
            loss_kl = torch.tensor(0).to(self.args.device)
        else:
            self.kliter += 1
            if "icod-icod-l,lg,lv,ckl":
                cycle_num = (self.args.total_training_steps/4)
                mod = self.kliter % cycle_num
                temp = mod/cycle_num
                beta = 1
                if temp <= r:
                    beta = 1/(1 + np.exp(-temp))

                loss_kl = kld
                total_loss = loss + beta * kld
            if "icod-icod-l,lg,lv,akl":
                kl_weight = min(math.tanh(6 * self.kliter /
                                          self.args.full_kl_step - 3) + 1, 1)
                loss_kl = self.args.kl_ceiling * kl_weight + kld
                total_loss = loss_rec + loss_kl

        return total_loss, loss_rec, loss_kl

    def training_step(self, batch, batch_idx):
        if self.args.num_warmup_steps == self.iter:
            self.latent_transformer = True
            self.model.model.switch_latent_transformer(self.latent_transformer)
            self.configure_optimizers()  # reset the momentum

        loss, kld = self(batch)
        total_loss, loss_rec, loss_kl = self.calculate_losses(loss, kld)
        self.log('total train loss', total_loss)
        self.log('rec train loss', loss_rec)
        self.log('kl train loss', loss_kl)
        self.iter += 1
        return total_loss

    def validation_step(self, batch, batch_idx):
        return batch

    def validation_epoch_end(self, batch):
        print("##### End of Epoch validation #####")

        batch = batch[0]

        loss, kld = self(batch, val=True)
        total_loss, loss_rec, loss_kl = self.calculate_losses(loss, kld)
        self.log('total val loss', total_loss)
        self.log('rec val loss', loss_rec)
        self.log('kl val loss', loss_kl)
        self.val_losses["total loss"].append(total_loss.item())
        self.val_losses["rec loss"].append(loss_rec.item())
        self.val_losses["kl loss"].append(loss_kl.item())

        # scores = self.decode_and_print(batch)
        # for k, v in scores.items():
        #     rounded_val = np.round(np.mean(v) * 100, 4)
        #     self.log("val_"+k, rounded_val)
        #     print(k, "\t", rounded_val)

        print("********** DECODING OBJECT SELECTED QUESTIONS ********** ")
        qa_decode_scores = self.decode_and_print(batch, qa_decode=True)
        for k, v in qa_decode_scores.items():
            rounded_val = np.round(np.mean(v) * 100, 4)
            self.log("val_qa_decode_"+k, rounded_val)
            print(k, "\t", rounded_val)
        print("*"*20)
#
        for k, v in self.val_losses.items():
            print("val", k + ":", np.round(np.mean(v), 4))
            self.val_losses[k] = []

        print()
        print("This was validating after iteration {}".format(self.iter))

    def test_step(self, batch, batch_idx):
        scores = self.decode_and_print(batch, qa_decode=True, val=False)
        for k, v in scores.items():
            rounded_val = np.round(np.mean(v) * 100, 4)
            print(k, "\t", rounded_val)

        for k, v in scores.items():
            if k not in self.test_scores.keys():
                self.test_scores[k] = []
            else:
                self.test_scores[k].append(v)

        return scores

    def test_end(self, all_scores):
        for k, scores in self.test_scores.items():
            self.test_scores[k] = np.mean(self.test_scores[k])
        return all_scores

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def decode_and_print(self, batch, qa_decode=False, print_lim=20, val=True):

        images = batch["images"]
        image_ids = batch["image_ids"]
        question_ids = batch["question_ids"]
        object_features = batch["object_features"]
        object_locations = batch["object_locations"]

        inference_ids = batch["legal_ids"]
        inference_attention_masks = batch["legal_attention_masks"]

        if qa_decode:
            inference_ids = batch["qa_inference_ids"]
            inference_attention_masks = batch["qa_inference_attention_masks"]

        if self.args.variant in ("ifD-ifD"):
            inference_ids = batch["caption_ids"]
            inference_attention_masks = batch["caption_attention_masks"]

        if self.args.variant in ("icf-icf"):
            inference_ids = batch["category_only_ids"]
            inference_attention_masks = batch["category_only_attn_masks"]

        decoded_questions = [self.tokenizer.decode(to_decode) for to_decode in question_ids]
        decoded_inputs = [self.tokenizer.decode(to_decode) for to_decode in inference_ids]

        preds = []
        gts = []
        decoded_sentences = self.model.decode_greedy(images, inference_ids, inference_attention_masks, object_features, object_locations)
        for i, sentence in enumerate(decoded_sentences):
            curr_input = self.filter_special_tokens(decoded_inputs[i])
            generated_q = self.filter_special_tokens(sentence)
            real_q = self.filter_special_tokens(decoded_questions[i])

            gts.append(real_q)
            preds.append(generated_q)

            if i < print_lim:
                print("Image ID:\t", image_ids[i])
                if not self.args.variant in ("ifD-ifD"):
                    print("Category:\t", curr_input.split()[0])
                    print("KW inputs:\t", " ".join(curr_input.split()[1:]))
                else:
                    print("Caption:", " ".join(curr_input.split()))
                print("Generated:\t", generated_q)
                print("Real Ques:\t", real_q)
                print()

        scores = self.nlge.compute_metrics(ref_list=[gts], hyp_list=preds)

        msd = MultisetDistances(references=gts)
        msj_distance = msd.get_jaccard_score(sentences=preds)
        new_msj_distance = {}
        for k in msj_distance.keys():
            new_msj_distance["msj_{}".format(k)] = msj_distance[k]
        scores.update(new_msj_distance)

        if val:
            for k, v in scores.items():
                rounded_val = np.round(np.mean(v) * 100, 4)
                if k == "Bleu_4":
                    self.bleus.append((self.iter, rounded_val))
                elif k == "msj_4":
                    self.msjs.append((self.iter, rounded_val))
                elif k == "fbd":
                    self.fbds.append((self.iter, rounded_val))

        max_bleu = max(self.bleus, key=itemgetter(1))
        max_msjs = max(self.msjs, key=itemgetter(1))
        # min_fbds = min(self.fbds, key=itemgetter(1))
        print("HIGHEST BLEU SCORE WAS: {} FROM ITER {}".format(
            max_bleu[1], max_bleu[0]))
        print("HIGHEST MSJ_4 SCORE WAS: {} FROM ITER {}".format(
            max_msjs[1], max_msjs[0]))
        # print("SMALLEST FBD SCORE WAS: {} FROM ITER {}".format(min_fbds[1], min_fbds[0]))

        print("Model Variant:", self.args.variant)

        return scores

    def filter_special_tokens(self, decoded_sentence_string):
        decoded_sentence_list = decoded_sentence_string.split()
        special_tokens = self.tokenizer.all_special_tokens

        if self.tokenizer.sep_token in decoded_sentence_list:
            index_of_end = decoded_sentence_list.index(self.tokenizer.sep_token)
            decoded_sentence_list = decoded_sentence_list[:index_of_end]

        filtered = []
        for token in decoded_sentence_list:
            if token not in special_tokens:
                filtered.append(token)
        return " ".join(filtered)


class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        if pl_module.iter > pl_module.args.num_warmup_steps:
            self._run_early_stopping_check(trainer, pl_module)


early_stop_callback = EarlyStopping(
    monitor='val_qa_decode_Bleu_4',
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode='max'
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="Hidden dimensionality of the model")
    parser.add_argument("--latent_dim", type=int, default=768,
                        help="Hidden dimensionality of the model")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate of the network")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_warmup_steps", type=float, default=2000,
                        help="Number of warmup steps before turning on latent transformer")
    parser.add_argument("--total_training_steps", type=int, default=35000,
                        help="Total number of training steps for the model")
    parser.add_argument("--kl_ceiling", type=int, default=0.5)
    parser.add_argument("--full_kl_step", type=int, default=3000, help="Total steps until kl is fully annealed")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers in encoder and decoder")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of heads in the multi-head attention")
    parser.add_argument("--max_decode_len", type=int, default=50, help="Maximum length decoded sequences are allowed to be")
    parser.add_argument("--variant", type=str, default="icod-icod", help="Model variant to run.")
    parser.add_argument("--dataset", type=str,
                        default="/data/nv419/VQG_DATA/processed/iq_dataset.hdf5")
    parser.add_argument("--val_dataset", type=str,
                        default="/data/nv419/VQG_DATA/processed/iq_val_dataset.hdf5")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)],)

    data_loader = get_loader(os.path.join(
        os.getcwd(), args.dataset), tokenizer, args.batch_size, shuffle=True, num_workers=8)
    val_data_loader = get_loader(os.path.join(
        os.getcwd(), args.val_dataset), tokenizer, args.batch_size, shuffle=False, num_workers=8)

    trainVQG = TrainVQG(args, tokenizer)  # .to(device)

    trainer = pl.Trainer(max_steps=args.total_training_steps, gradient_clip_val=5,
                         val_check_interval=500, limit_val_batches=400, callbacks=[early_stop_callback], gpus=1)

    trainer.fit(trainVQG, data_loader, val_data_loader)

    test_data_loader = get_loader(os.path.join(
        os.getcwd(), args.val_dataset), tokenizer, args.batch_size, shuffle=False, num_workers=8)
    trainer.test(trainVQG, test_dataloaders=test_data_loader, ckpt_path="best")

    for k, scores in trainVQG.test_scores.items():
        print(k, np.mean(scores))
