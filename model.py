from layers import ImageTransformerEncoder, LatentNorm
from torch import nn
import torch
from transformers import BertModel, BertLMHeadModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.bert.tokenization_bert import BertTokenizer


class VQGModel(nn.Module):
    def __init__(self, args, tokenizer: BertTokenizer, latent_transformer) -> None:
        super().__init__()

        self.args = args

        self.image_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768, momentum=0.01)
        )

        config = BertConfig.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(config)

        if self.args.variant.split("-")[0] == "icodqa":
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        if self.args.variant == "icodqaf-icodqaf":
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.multi_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.image_transformer = ImageTransformerEncoder(args)

        self.decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True, use_cache=True, add_cross_attention=True)
        self.latent_transformer = latent_transformer
        self.latent = LatentNorm(args)

    def switch_latent_transformer(self, new_mode):
        self.latent_transformer = new_mode

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations):
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

        if self.args.variant == "icodqaf-icodqaf":
            encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
            encoded_objects = encoded_objects.permute(1, 0, 2)
            object_mask = torch.ones(bsz, 36).long().to(self.args.device)
            encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)

            token_type_ids_text = torch.zeros_like(encoder_attention_mask).long().to(self.args.device)  # [B, ~13]
            token_type_ids_objs = torch.ones_like(object_mask).long().to(self.args.device)  # [B, 36]
            token_type_ids = torch.cat((token_type_ids_text, token_type_ids_objs), dim=1)
            multi_pos_ids = torch.cat((encoder_pos_ids, torch.zeros(bsz, 36).long().to(self.args.device)), dim=1)

            multi_encoder_inputs = torch.cat((encoder_hidden_states, encoded_objects), dim=1)
            multi_encoder_outputs = self.multi_encoder(inputs_embeds=multi_encoder_inputs, position_ids=multi_pos_ids, attention_mask=src_mask, token_type_ids=token_type_ids)
            encoder_hidden_states = multi_encoder_outputs.last_hidden_state

        kld = None
        if self.latent_transformer:
            encoder_hidden_states, kld = self.latent(encoder_hidden_states)
            # encoder_hidden_states[:, 0] = encoder_hidden_states[:, 0] + z  # Why doesn't this work??
            # first_encoder_hidden_state = encoder_hidden_states[:, 0] + z
            # encoder_hidden_states = torch.cat((first_encoder_hidden_state.unsqueeze(1), encoder_hidden_states[:, 1:]), dim=1)

        outputs = self.decoder(input_ids=question_ids, labels=question_ids, attention_mask=question_attention_masks,
                               encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        loss = outputs["loss"]

        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, obj_features, obj_locations, max_decode_len=50):
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

        if self.args.variant == "icodqaf-icodqaf":
            encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
            encoded_objects = encoded_objects.permute(1, 0, 2)
            object_mask = torch.ones(bsz, 36).long().to(self.args.device)
            encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)

            token_type_ids_text = torch.zeros_like(encoder_attention_mask).long().to(self.args.device)  # [B, ~13]
            token_type_ids_objs = torch.ones_like(object_mask).long().to(self.args.device)  # [B, 36]
            token_type_ids = torch.cat((token_type_ids_text, token_type_ids_objs), dim=1)
            multi_pos_ids = torch.cat((encoder_pos_ids, torch.zeros(bsz, 36).long().to(self.args.device)), dim=1)

            multi_encoder_inputs = torch.cat((encoder_hidden_states, encoded_objects), dim=1)
            multi_encoder_outputs = self.multi_encoder(inputs_embeds=multi_encoder_inputs, position_ids=multi_pos_ids, attention_mask=src_mask, token_type_ids=token_type_ids)
            encoder_hidden_states = multi_encoder_outputs.last_hidden_state

        kld = None
        if self.latent_transformer:
            encoder_hidden_states, kld = self.latent(encoder_hidden_states)
            # encoder_hidden_states[:, 0] = encoder_hidden_states[:, 0] + z  # Why doesn't this work??
            # first_encoder_hidden_state = encoder_hidden_states[:, 0] + z
            # encoder_hidden_states = torch.cat((first_encoder_hidden_state.unsqueeze(1), encoder_hidden_states[:, 1:]), dim=1)

        ys = torch.ones(images.shape[0], 1).fill_(self.tokenizer.cls_token_id).long().to(self.args.device)

        for i in range(max_decode_len):
            output = self.decoder(input_ids=ys, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            logits = output["logits"]
            token = torch.argmax(logits[:, -1], dim=1)

            ys = torch.cat((ys, token.unsqueeze(1)), dim=1)

        sequences = []
        for batch in ys:
            sequence = self.tokenizer.decode(batch)
            sequences.append(sequence)

        return sequences
