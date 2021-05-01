from layers import ImageTransformerEncoder
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLMHeadModel, BertModel


class BaseVQG(nn.Module):
    def __init__(self, args, tokenizer, object_features_variant=False, latent_transformer=False):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer

        self.image_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768, momentum=0.01)
        )

        config = BertConfig.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(config)

        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True, use_cache=True, add_cross_attention=True)

        if object_features_variant:
            self.image_transformer = ImageTransformerEncoder(args)

        self.latent_transformer = latent_transformer

    def switch_latent_transformer(self, new_mode):
        self.latent_transformer = new_mode

    def forward(self, *args):
        pass

    def encode_image_and_text(self, images, input_ids, input_attention_masks, question_ids, return_embeddings=True):
        bsz = images.shape[0]

        images = self.image_projection(images).unsqueeze(1)  # [B, 1, D]

        position_ids = torch.zeros_like(input_ids).long().to(self.args.device)  # [B, T]
        embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids
        )  # [B, T, D]

        image_pos_id = torch.zeros(bsz, 1).long().to(self.args.device)  # [B, 1]
        encoder_pos_ids = torch.cat((image_pos_id, position_ids), dim=1)

        image_pad_mask = torch.ones(bsz, 1).long().to(self.args.device)
        encoder_attention_mask = torch.cat((image_pad_mask, input_attention_masks), dim=1)

        encoder_inputs = torch.cat((images, embedding), dim=1)
        encoder_outputs = self.text_encoder(inputs_embeds=encoder_inputs, position_ids=encoder_pos_ids, attention_mask=encoder_attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, T+1, D]

        target_embedding = None
        if return_embeddings:
            target_pos_ids_single = torch.arange(question_ids.shape[-1]).to(self.args.device)  # [T_q]
            target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_q]
            target_embedding = self.embeddings(
                input_ids=question_ids,
                position_ids=target_pos_ids_batch
            )  # [B, T_q, D]

        return encoder_hidden_states, encoder_attention_mask, target_embedding

    def encode_object_features(self, obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask):
        bsz = obj_features.shape[0]
        encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
        encoded_objects = encoded_objects.permute(1, 0, 2)
        object_mask = torch.ones(bsz, 36).long().to(self.args.device)
        encoder_hidden_states = torch.cat((encoder_hidden_states, encoded_objects), dim=1)
        encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)
        return encoder_hidden_states, encoder_attention_mask

    def forward_decode(self, target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask):
        outputs = self.decoder(inputs_embeds=target_embedding, labels=question_ids, attention_mask=question_attention_masks,
                               encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        loss = outputs["loss"]

        return loss

    def decode_greedy_hidden_states(self, images, input_ids, input_attention_masks):
        encoder_hidden_states, encoder_attention_mask, _ = self.encode_image_and_text(images, input_ids, input_attention_masks, question_ids=None, return_embeddings=False)
        return encoder_hidden_states, encoder_attention_mask

    def decode_greedy_obj_features(self, obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask):
        return self.encode_object_features(obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask)

    def decode_greedy_sequence(self, encoder_hidden_states, encoder_attention_mask, z=0):
        batch_size = encoder_hidden_states.shape[0]
        ys = torch.ones(batch_size, 1).fill_(self.tokenizer.cls_token_id).long().to(self.args.device)
        for _ in range(self.args.max_decode_len):
            target_pos_ids_single = torch.arange(ys.shape[-1]).to(self.args.device)  # [T_ys]
            target_pos_ids_batch = target_pos_ids_single.repeat(batch_size, 1)  # [B, T_ys]
            target_embedding = self.embeddings(
                input_ids=ys,
                position_ids=target_pos_ids_batch
            )  # [B, T_q, D]

            target_embedding[:, 0] = target_embedding[:, 0] + z

            output = self.decoder(inputs_embeds=target_embedding, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            logits = output["logits"]
            token = torch.argmax(logits[:, -1], dim=1)

            ys = torch.cat((ys, token.unsqueeze(1)), dim=1)

        sequences = []
        for batch in ys:
            sequence = self.tokenizer.decode(batch)
            sequences.append(sequence)

        return sequences
