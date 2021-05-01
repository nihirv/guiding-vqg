from layers import Latent
from variants.base import BaseVQG
from transformers.models.bert.modeling_bert import BertModel


class icod_icod_l(BaseVQG):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        self.generative_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.latent_layer = Latent(args)

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, *args):
        encoder_hidden_states, encoder_attention_mask, target_embedding = self.encode_image_and_text(images, input_ids, input_attention_masks, question_ids)

        generative_outputs = self.generative_encoder(input_ids=question_ids, attention_mask=question_attention_masks)
        generative_hidden_states = generative_outputs.last_hidden_state  # [B, T_q, D]
        kld = None
        if self.latent_transformer:
            z, kld = self.latent_layer(encoder_hidden_states[:, 0], generative_hidden_states[:, 0])
            target_embedding[:, 0] = target_embedding[:, 0] + z

        loss = self.forward_decode(target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask)
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, *args):
        encoder_hidden_states, encoder_attention_mask = self.decode_greedy_hidden_states(images, input_ids, input_attention_masks)

        z = 0
        if self.latent_transformer:
            z, _ = self.latent_layer(encoder_hidden_states[:, 0], None)

        sequences = self.decode_greedy_sequence(encoder_hidden_states, encoder_attention_mask, z=z)
        return sequences
