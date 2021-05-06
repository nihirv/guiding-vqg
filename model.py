from torch.tensor import Tensor
from torch import nn


class VQGModel(nn.Module):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()

        if args.variant in ("icod-icod", "icodqa-icodqa"):
            from variants.icod_icod import icod_icod
            self.model = icod_icod(args, tokenizer)

        if args.variant in ("icodqaf-icodqaf", "icodqaf-icof", "icodf-icodf", "icf-icf"):
            from variants.icodf_icodf import icodf_icodf
            self.model = icodf_icodf(args, tokenizer)
        if args.variant in ("ifD-ifD"):
            from variants.icodf_icodf import icodf_icodf
            self.model = icodf_icodf(args, tokenizer, positional_embed_variant=True)

        if args.variant in ("icod-icod-l,lg,lv,ckl", "icod-icod-l,lg,lv,akl"):
            from variants.icod_icod_l import icod_icod_l
            self.model = icod_icod_l(args, tokenizer)

        if args.variant in ("icod-icod--lstm"):
            from variants.icod_icod__lstm import icod_icod__lstm
            self.model = icod_icod__lstm(args, tokenizer)

        if args.variant in ("if-if"):
            from variants.if_if import if_if
            self.model = if_if(args, tokenizer)

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations):
        loss, kld = self.model(images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations)
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, obj_features, obj_locations):
        return self.model.decode_greedy(images, input_ids, input_attention_masks, obj_features, obj_locations)
