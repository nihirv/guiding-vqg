from layers import Decoder
from variants.base import BaseVQG
from torch import nn
import torch
import random


class icod_icod__lstm(BaseVQG):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)

        self.decoder = Decoder(args, len(tokenizer))
        self.criterion = nn.CrossEntropyLoss()

    def decode_rnn_training(self, encoder_hidden_states, z, target, teacher_forcing_ratio=0.5):
        outputs = torch.zeros(
            target.shape[1], target.shape[0], len(self.tokenizer), device=self.args.device)
        input = target[:, 0]
        hidden = z.contiguous()

        for t in range(1, target.shape[1]):
            output, hidden = self.decoder(
                input, hidden, encoder_hidden_states.contiguous())
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1

        return outputs.permute(1, 0, 2)  # [B, T, D]

    def ce_loss_for_rnn(self, pred_logits, target):
        loss_rec = self.criterion(
            pred_logits[1:].reshape(-1, pred_logits.size(-1)), target[1:].reshape(-1))
        return loss_rec

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, *args):
        encoder_hidden_states, _, _ = self.encode_image_and_text(images, input_ids, input_attention_masks, question_ids)
        decoder_outputs = self.decode_rnn_training(
            encoder_hidden_states, encoder_hidden_states[:, 0], question_ids
        )
        loss = self.ce_loss_for_rnn(decoder_outputs, question_ids)
        kld = None
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, *args):
        encoder_hidden_states, _ = self.decode_greedy_hidden_states(images, input_ids, input_attention_masks)
        ys = torch.ones(images.shape[0], 1).fill_(self.tokenizer.cls_token_id).long().to(self.args.device)

        input = ys[:, 0]
        hidden = encoder_hidden_states[:, 0].contiguous()

        for _ in range(self.args.max_decode_len):
            logits, hidden = self.decoder(
                input, hidden, encoder_hidden_states)
            input = torch.argmax(logits, dim=-1)
            ys = torch.cat((ys, input.unsqueeze(1)), dim=1)

        sequences = []
        for batch in ys:
            sequence = self.tokenizer.decode(batch)
            sequences.append(sequence)

        return sequences
