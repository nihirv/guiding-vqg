from variants.base import BaseVQG


class icod_icod(BaseVQG):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, *args):
        encoder_hidden_states, encoder_attention_mask, target_embedding = self.encode_image_and_text(images, input_ids, input_attention_masks, question_ids)
        loss = self.forward_decode(target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask)
        kld = None
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, *args):
        encoder_hidden_states, encoder_attention_mask = self.decode_greedy_hidden_states(images, input_ids, input_attention_masks)
        sequences = self.decode_greedy_sequence(encoder_hidden_states, encoder_attention_mask)
        return sequences
