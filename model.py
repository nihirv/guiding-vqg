from layers import ImageTransformerEncoder, Latent, LatentNorm
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

        if self.args.variant.split("-")[0] in ("icod", "icodqa"):
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        if self.args.variant in ("icodqaf-icodqaf", "icodqaf-icof"):
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.multi_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.image_transformer = ImageTransformerEncoder(args)
        if self.args.variant == "icod-icod-l,lg,lv":
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.generative_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
            self.latent_layer = Latent(args)

        self.decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True, use_cache=True, add_cross_attention=True)
        self.latent_transformer = latent_transformer

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

        kld = None

        # We'll manually construct the target embeddings, because in the case of a latent variable, we want to add some features to the embedding
        target_pos_ids_single = torch.arange(question_ids.shape[-1]).to(self.args.device)  # [T_q]
        target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_q]
        target_embedding = self.embeddings(
            input_ids=question_ids,
            position_ids=target_pos_ids_batch
        )  # [B, T_q, D]

        if self.args.variant == "icodqaf-icodqaf" or self.args.variant == "icodqaf-icof":
            encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
            encoded_objects = encoded_objects.permute(1, 0, 2)
            object_mask = torch.ones(bsz, 36).long().to(self.args.device)
            encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)
            encoder_hidden_states = torch.cat((encoder_hidden_states, encoded_objects), dim=1)

        if self.args.variant == "icod-icod-l,lg,lv":
            generative_outputs = self.generative_encoder(input_ids=question_ids, attention_mask=question_attention_masks)
            generative_hidden_states = generative_outputs.last_hidden_state  # [B, T_q, D]
            if self.latent_transformer:
                z, kld = self.latent_layer(encoder_hidden_states[:, 0], generative_hidden_states[:, 0])
                target_embedding[:, 0] = target_embedding[:, 0] + z

        outputs = self.decoder(inputs_embeds=target_embedding, labels=question_ids, attention_mask=question_attention_masks,
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

        if self.args.variant in ("icodqaf-icodqaf", "icodqaf-icof"):
            encoded_objects, _ = self.image_transformer(obj_features, obj_locations)  # [B, 36, D]
            encoded_objects = encoded_objects.permute(1, 0, 2)
            object_mask = torch.ones(bsz, 36).long().to(self.args.device)
            encoder_attention_mask = torch.cat((encoder_attention_mask, object_mask), dim=1)
            encoder_hidden_states = torch.cat((encoder_hidden_states, encoded_objects), dim=1)

        z, kld = torch.tensor(0).to(self.args.device), None
        if self.args.variant == "icod-icod-l,lg,lv":
            if self.latent_transformer:
                z, kld = self.latent_layer(encoder_hidden_states[:, 0], None)

        ys = torch.ones(images.shape[0], 1).fill_(self.tokenizer.cls_token_id).long().to(self.args.device)

        for i in range(max_decode_len):
            target_pos_ids_single = torch.arange(ys.shape[-1]).to(self.args.device)  # [T_ys]
            target_pos_ids_batch = target_pos_ids_single.repeat(bsz, 1)  # [B, T_ys]
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
