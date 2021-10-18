from layers import Latent
from variants.base import BaseVQG
from transformers.models.bert.modeling_bert import BertModel
# from transformers.models.bert.modeling_bert import 

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.weight_norm import weight_norm


class icod_icod_l(BaseVQG):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer)
        # self.obj_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.qa_encoder = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.latent_layer = Latent(args)

        # self.nonlinear1 = FCNet([768 + 768, args.hidden_dim])
        # self.linear1 = weight_norm(nn.Linear(args.hidden_dim, 2), dim=None)
        self.nonlinear2 = FCNet([768 + 768, args.hidden_dim])
        self.linear2 = weight_norm(nn.Linear(args.hidden_dim, 1), dim=None)

        # self.var_attention = NewAttention(768, 768, 128, 3)
        # self.gen_attention = NewAttention(768, 768, 128, 3)

        self.logit_fc = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(args.hidden_dim * 2, eps=1e-12),
            nn.Linear(args.hidden_dim * 2, args.num_categories)
        )

        self.category_loss = nn.CrossEntropyLoss()
        self.cat_to_hidden = nn.Linear(16, args.hidden_dim)

        for param in self.qa_encoder.parameters():
            param.requires_grad = False

    def forward(self, images, question_ids, question_attention_masks, input_ids, input_attention_masks, obj_features, obj_locations, object_embeddings=None, target_category=None):
        bs = images.size()[0]

        # print(obj_features)
        # print(object_embeddings)
        # gen_repr = torch.cat((obj_features, object_embeddings), dim=2)
        # gen_repr = self.linear1(self.nonlinear1(gen_repr)).view(bs, -1)
        # gen_sigmoid = torch.sigmoid(gen_repr)
        # gen_binary = gen_sigmoid >= 0.5
        # qa_output = self.qa_encoder(input_ids=input_ids, attention_mask=input_attention_masks).last_hidden_state
        # # var_repr = self.var_attention(object_embeddings, qa_output[:, 0]).squeeze()
        # num_objs = obj_features.size(1)
        # qa_repr = qa_output[:, 0].unsqueeze(1).repeat(1, num_objs, 1)
        # qa_repr = torch.cat((object_embeddings, qa_repr), dim=2)
        # qa_repr = self.linear2(self.nonlinear2(qa_repr)).squeeze()
        # z = F.softmax(qa_repr, dim=1).unsqueeze(2)

        
        # object_embeddings = object_embeddings * z

        encoder_hidden_states, encoder_attention_mask, target_embedding = self.encode_image_and_object(images, object_embeddings, question_ids)
        encoder_hidden_states, encoder_attention_mask, gen_repr = self.encode_object_features(obj_features, obj_locations, encoder_hidden_states, encoder_attention_mask)

        # gen_repr2 = self.gen_attention(object_embeddings, gen_repr).permute(0,2,1)
        # var_repr2 = self.var_attention(object_embeddings, qa_output[:, 0])
        # z, kld = self.latent_layer(gen_repr2, var_repr2)
        # z = gumbel_softmax(z, 1.0, hard=True)
        # z = z.sum(1).squeeze().clamp(0,1).unsqueeze(2)
        # print(var_repr2.shape)
        # exit()
        # z = var_repr2
        # object_embeddings = object_embeddings * z
        # object_embeddings = object_embeddings.mean(1).squeeze()

        logit_category = self.logit_fc(gen_repr)
        loss_cat = self.category_loss(logit_category, target_category)
        # # cat_one_hot = torch.zeros_like(logit_category).cuda().scatter_(1, target_category.view(bs, -1), 1)
        z_cat = self.cat_to_hidden(logit_category)
        target_embedding[:, 0] = target_embedding[:, 0] + z_cat

        loss = self.forward_decode(target_embedding, question_ids, question_attention_masks, encoder_hidden_states, encoder_attention_mask)
        loss += loss_cat
        kld = None
        return loss, kld

    def decode_greedy(self, images, input_ids, input_attention_masks, object_embeddings, obj_features, object_locations, target_category=None):
        bs = images.size()[0]

        qa_output = self.qa_encoder(input_ids=input_ids, attention_mask=input_attention_masks).last_hidden_state
        # # var_repr = self.var_attention(object_embeddings, qa_output[:, 0]).squeeze()
        num_objs = obj_features.size(1)
        qa_repr = qa_output[:, 0].unsqueeze(1).repeat(1, num_objs, 1)
        qa_repr = torch.cat((object_embeddings, qa_repr), dim=2)
        qa_repr = self.linear2(self.nonlinear2(qa_repr)).squeeze()
        z = F.softmax(qa_repr, dim=1).unsqueeze(2)
        # # print(z.squeeze()[0])
        _, max_obj = z.squeeze().topk(20, dim=1)
        object_embeddings = object_embeddings * z

        encoder_hidden_states, encoder_attention_mask = self.decode_greedy_hidden_states(images, object_embeddings)
        encoder_hidden_states, encoder_attention_mask, gen_repr = self.decode_greedy_obj_features(obj_features, object_locations, encoder_hidden_states, encoder_attention_mask)

        # gen_repr2 = self.gen_attention(object_embeddings, gen_repr).permute(0,2,1)
        # var_repr2 = self.var_attention(object_embeddings, qa_output[:, 0])
        # z, _ = self.latent_layer(gen_repr2)
        # z = gumbel_softmax(z, 1.0, hard=True)
        # z = z.sum(1).squeeze().clamp(0,1).unsqueeze(2)
        # z = var_repr2
        # object_embeddings = object_embeddings * z
        # object_embeddings = object_embeddings.mean(1).squeeze()

        logit_category = self.logit_fc(gen_repr)
        max_cat = torch.argmax(logit_category, dim=1).view(-1)
        # # pred_cat_one_hot = torch.zeros_like(logit_category).cuda().scatter_(1, target_category.view(bs, -1), 1)
        z_cat = self.cat_to_hidden(logit_category)

        sequences = self.decode_greedy_sequence(encoder_hidden_states, encoder_attention_mask, z=0)
        return sequences, max_obj, max_cat


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, out_dim, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(num_hid, out_dim), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        # w = nn.functional.softmax(logits, 1)

        return logits

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q)
        if q.size()[1] != k:
            q_proj = q_proj.unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, 3, 36)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, 3, 36)
