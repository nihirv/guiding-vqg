from torch.tensor import Tensor
from layers import Decoder, ImageTransformerEncoder, Latent, LatentNorm
from torch import nn
import torch
from transformers import BertModel, BertLMHeadModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.bert.tokenization_bert import BertTokenizer
import random


class VQGModel(nn.Module):
    def __init__(self, args, tokenizer: BertTokenizer) -> None:
        super().__init__()

        if self.args.variant in ("icod-icod", "icodqa-icodqa"):
            from variants.icod_icod import icod_icod
            self.model = icod_icod(args, tokenizer)

        if self.args.variant in ("icodqaf-icodqaf", "icodqaf-icof", "icodf-icodf"):
            from variants.icodf_icodf import icodf_icodf
            self.model = icodf_icodf(args, tokenizer)

        if self.args.variant in ("icod-icod-l,lg,lv,ckl", "icod-icod-l,lg,lv,akl"):
            from variants.icod_icod_l import icod_icod_l
            self.model = icod_icod_l(args, tokenizer)

        if self.args.variant in ("icod-icod--lstm"):
            from variants.icod_icod__lstm import icod_icod__lstm
            self.model = icod_icod__lstm(args, tokenizer)

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations):
        loss, kld = self.model(images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations)
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, obj_features, obj_locations):
        return self.model.decode_greedy(images, input_ids, input_attention_masks, obj_features, obj_locations)
