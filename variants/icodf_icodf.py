from operator import pos
from variants.base import BaseVQG


class icodf_icodf(BaseVQG):
    def __init__(self, args, tokenizer, positional_embed_variant=False) -> None:
        super().__init__(args, tokenizer, object_features_variant=True, positional_embed_variant=positional_embed_variant)

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations, *args):
        encoder_hidden_states, encoder_attention_mask, target_embedding = self.encode_image_and_text(images, input_ids, input_attention_masks, question_ids)
        encoder_hidden_states, encoder_attention_mask = self.encode_object_features(obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask)
        loss = self.forward_decode(target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask)
        kld = None
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, obj_features, obj_locations, *args):
        encoder_hidden_states, encoder_attention_mask = self.decode_greedy_hidden_states(images, input_ids, input_attention_masks)
        encoder_hidden_states, encoder_attention_mask = self.decode_greedy_obj_features(obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask)
        sequences = self.decode_greedy_sequence(encoder_hidden_states, encoder_attention_mask)
        return sequences
