import torch
from variants.base import BaseVQG


class if_if(BaseVQG):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer, object_features_variant=True)

        self.args = args

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations):
        bsz = obj_features.shape[0]
        target_pos_ids_single = torch.arange(question_ids.shape[-1]).to(self.args.device)  # [T_q]
        target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_q]
        target_embedding = self.embeddings(
            input_ids=question_ids,
            position_ids=target_pos_ids_batch)

        encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
        encoded_objects = encoded_objects.permute(1, 0, 2)
        object_mask = torch.ones(bsz, 36).long().to(self.args.device)
        loss = self.forward_decode(target_embedding, question_ids, question_attention_masks, encoded_objects, object_mask)
        kld = None
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, obj_features, obj_locations):
        bsz = obj_features.shape[0]
        # print(obj_features,obj_features.shape)
        encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
        encoded_objects = encoded_objects.permute(1, 0, 2)
        object_mask = torch.ones(bsz, 36).long().to(self.args.device)
        encoder_hidden_states, encoder_attention_mask, _ = self.decode_greedy_obj_features(obj_features, obj_locations, encoded_objects, object_mask)
        sequences = self.decode_greedy_sequence(encoder_hidden_states, encoder_attention_mask)
        return sequences
