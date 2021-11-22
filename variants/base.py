import copy
from typing import Union
from transformers.models.bert.tokenization_bert import BertTokenizer
from layers import ImageTransformerEncoder
import torch
import torch.nn as nn
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLMHeadModel, BertModel, BertEncoder


class BaseVQG(nn.Module):
    def __init__(self, args, tokenizer: BertTokenizer, object_features_variant=False, positional_embed_variant=False, latent_transformer=False):
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

        encoder_config = BertConfig.from_pretrained("bert-base-uncased", return_dict=True)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", config=encoder_config)

        decoder_config = BertConfig.from_pretrained(
            "bert-base-uncased", is_decoder=True, use_cache=True, add_cross_attention=True
        )
        self.decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=decoder_config)

        # if object_features_variant:
        self.image_transformer = ImageTransformerEncoder(args)

        self.positional_embed = True if positional_embed_variant else False

        self.latent_transformer = latent_transformer

        if self.args.truncate:
            print("TRUNCATING (BASE)")
            self.text_encoder, src_config = self.truncate_model(self.text_encoder, encoder_config)
            self.decoder, trg_config = self.truncate_model(self.decoder, decoder_config)

    def truncate_model(self, model: Union[BertModel, BertLMHeadModel], config: BertConfig):
        if config.is_decoder:
            extracted_layers = [model.bert.encoder.layer[0], model.bert.encoder.layer[-1]]
        else:
            extracted_layers = [model.encoder.layer[0], model.encoder.layer[-1]]
        config.num_hidden_layers = 2
        new_model = copy.deepcopy(model)
        new_encoder = BertEncoder(config)
        new_encoder.layer = nn.ModuleList(extracted_layers)
        if config.is_decoder:
            new_model.bert.encoder = new_encoder
        else:
            new_model.encoder = new_encoder
        return new_model, config

    def switch_latent_transformer(self, new_mode):
        self.latent_transformer = new_mode

    def forward(self, *args):
        pass

    def encode_image_and_text(self, images, input_ids, input_attention_masks, question_ids, return_target_embeddings=True):
        bsz = images.shape[0]

        images = self.image_projection(images).unsqueeze(1)  # [B, 1, D]

        if not self.positional_embed:
            # our default use case... we'll disable positional embedding because we're inputting a set of tokens, not a sequence
            position_ids = torch.zeros_like(input_ids).long().to(self.args.device)  # [B, T]
        else:
            # if position_ids = None, BERTEmbeddings will automatically apply position embeddings for us
            position_ids = None

        embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids
        )  # [B, T, D]

        if not self.positional_embed:
            # if we're working with our default set -> sequence setting...
            # we disable pos_enc by setting all position_ids to zero
            image_pos_id = torch.zeros(bsz, 1).long().to(self.args.device)  # [B, 1]
            encoder_pos_ids = torch.cat((image_pos_id, position_ids), dim=1)
        else:
            encoder_pos_ids = None

        image_pad_mask = torch.ones(bsz, 1).long().to(self.args.device)
        encoder_attention_mask = torch.cat((image_pad_mask, input_attention_masks), dim=1)

        encoder_inputs = torch.cat((images, embedding), dim=1)
        encoder_outputs = self.text_encoder(inputs_embeds=encoder_inputs, position_ids=encoder_pos_ids, attention_mask=encoder_attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, T+1, D]

        target_embedding = None
        if return_target_embeddings:
            target_pos_ids_single = torch.arange(question_ids.shape[-1]).to(self.args.device)  # [T_q]
            target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_q]
            target_embedding = self.embeddings(
                input_ids=question_ids,
                position_ids=target_pos_ids_batch
            )  # [B, T_q, D]

        return encoder_hidden_states, encoder_attention_mask, target_embedding

    def encode_image_and_object(self, images, object_embeddings, question_ids, return_target_embeddings=True):
        bsz = images.shape[0]
        images = self.image_projection(images).unsqueeze(1)  # [B, 1, D]

        # if not self.positional_embed:
        #     # our default use case... we'll disable positional embedding because we're inputting a set of tokens, not a sequence
        #     position_ids = torch.zeros_like(input_ids).long().to(self.args.device)  # [B, T]
        # else:
        #     # if position_ids = None, BERTEmbeddings will automatically apply position embeddings for us
        #     position_ids = None

        # embedding = self.embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids
        # )  # [B, T, D]
        embedding = object_embeddings
        image_pos_id = torch.zeros(bsz, 1).long().to(self.args.device)
        position_ids = torch.ones(bsz, self.args.latent_dim).long().to(self.args.device)
        encoder_pos_ids = torch.cat((image_pos_id, position_ids), dim=1)

        image_pad_mask = torch.ones(bsz, 1).long().to(self.args.device)
        input_attention_masks = torch.ones(bsz, self.args.latent_dim).long().to(self.args.device)
        encoder_attention_mask = torch.cat((image_pad_mask, input_attention_masks), dim=1)

        encoder_inputs = torch.cat((images, embedding), dim=1)
        encoder_outputs = self.text_encoder(inputs_embeds=encoder_inputs, position_ids=encoder_pos_ids, attention_mask=encoder_attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, T+1, D]

        target_embedding = None
        if return_target_embeddings:
            target_pos_ids_single = torch.arange(question_ids.shape[-1]).to(self.args.device)  # [T_q]
            target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_q]
            target_embedding = self.embeddings(
                input_ids=question_ids,
                position_ids=target_pos_ids_batch
            )  # [B, T_q, D]

        return encoder_hidden_states, encoder_attention_mask, target_embedding

    def encode_object_features(self, obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask):
        bsz = obj_features.shape[0]
        encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, self.args.latent_dim, D]
        encoded_objects = encoded_objects.permute(1, 0, 2)
        object_mask = torch.ones(bsz, self.args.latent_dim).long().to(self.args.device)
        encoder_hidden_states = torch.cat((encoder_hidden_states, encoded_objects), dim=1)
        encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)

        output = encoder_hidden_states[:, 0] + encoded_objects[:, 0]

        return encoder_hidden_states, encoder_attention_mask, output

    def forward_decode(self, target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask):
        outputs = self.decoder(inputs_embeds=target_embedding, labels=question_ids, attention_mask=question_attention_masks,
                               encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        loss = outputs["loss"]

        return loss

    def decode_greedy_hidden_states(self, images, object_embeddings):
        encoder_hidden_states, encoder_attention_mask, _ = self.encode_image_and_object(images, object_embeddings, question_ids=None, return_target_embeddings=False)
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
