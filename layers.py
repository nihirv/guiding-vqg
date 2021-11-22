import torch.nn.functional as F
import copy
from torch import nn
from fairseq.models import transformer
import torch


class Latent(nn.Module):
    def __init__(self, args, category_or_objects="objects", dropout=0):
        super(Latent, self).__init__()

        self.args = args
        if category_or_objects == "objects":
            self.latent_dim = self.args.latent_dim
            input_dim = self.args.latent_dim
        elif category_or_objects == "category":
            self.latent_dim = self.args.num_categories
            input_dim = self.args.hidden_dim

        latent_layer_network_architecture = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.latent_dim*2)
        )

        self.mean_logvar_prior = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            *copy.deepcopy(latent_layer_network_architecture)

        )
        self.mean_logvar_posterior = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            *copy.deepcopy(latent_layer_network_architecture)
        )

    def forward(self, x, x_p=None):
        mean_logvar_prior = self.mean_logvar_prior(x)
        mean_prior, logvar_prior = mean_logvar_prior[:, :self.latent_dim], mean_logvar_prior[:, self.latent_dim:]

        eps = torch.randn(mean_prior.size()).to(self.args.device)
        std = torch.exp(0.5 * logvar_prior)
        z = eps * std + mean_prior
        # z = mean_prior
        kld_loss = 0

        mean_posterior, logvar_posterior = None, None
        if x_p is not None:  # if x_p IS None, then we're in inference mode.
            mean_logvar_posterior = self.mean_logvar_posterior(x_p)
            mean_posterior, logvar_posterior = mean_logvar_posterior[:, :self.latent_dim], mean_logvar_posterior[:, self.latent_dim:]

            # for i in range(mean_posterior.size()[1]):
            #     kld_loss += gaussian_kld(mean_posterior[:, i], logvar_posterior[:, i], mean_prior[:, i], logvar_prior[:, i])
            # kld_loss = torch.mean(kld_loss)
            kld_loss += gaussian_kld(mean_posterior, logvar_posterior, mean_prior, logvar_prior)
            kld_loss = torch.mean(kld_loss)

            std = torch.exp(0.5 * logvar_posterior)
            z = eps * std + mean_posterior
        return z, kld_loss


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                           - torch.div(torch.pow(prior_mu - recog_mu,
                                                 2), torch.exp(prior_logvar))
                           - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), dim=-1)
    return kld


class LatentNorm(nn.Module):
    def __init__(self, args):
        super(LatentNorm, self).__init__()
        self.args = args

        self.latent_encoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(args.latent_dim*2, args.latent_dim*2)
        )

    def forward(self, hidden_states):
        latents = self.latent_encoder(hidden_states)
        mu, logvar = latents[:, :, :self.args.latent_dim], latents[:, :, self.args.latent_dim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.args.device)
        z = eps.mul(std).add_(mu)
        kld = gaussian_kld_norm(mu, logvar)
        return z, kld


def gaussian_kld_norm(mus, logvars, eps=1e-8):
    """Calculates KL distance of mus and logvars from unit normal.

    Args:
        mus: Tensor of means predicted by the encoder.
        logvars: Tensor of log vars predicted by the encoder.

    Returns:
        KL loss between mus and logvars and the normal unit gaussian.
    """
    KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    kl_loss = KLD/(mus.size(0) + eps)
    """
    if kl_loss > 100:
        print kl_loss
        print KLD
        print mus.min(), mus.max()
        print logvars.min(), logvars.max()
        1/0
    """
    return kl_loss


def create_padding_mask(src_tokens, src_lengths):
    padding_mask = torch.zeros(src_tokens.shape[:2],
                               dtype=torch.bool,
                               device=src_tokens.device)

    for i, src_length in enumerate(src_lengths):
        padding_mask[i, src_length:] = 1

    return padding_mask


class FeatureProjection(nn.Module):
    """
    Projects image features into a space of
    dimensionality `args.encoder_embed_dim`.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(2048, args.hidden_dim)

        # The following members are needed to
        # interface with TransformerEncoder.
        self.embedding_dim = args.hidden_dim
        self.padding_idx = -1

    def forward(self, x):
        return self.linear(x)


class ImageTransformerEncoder(transformer.TransformerEncoder):
    def __init__(self, args):
        args.encoder_layerdrop = args.dropout
        args.max_source_positions = 1000
        args.no_scale_embedding = False
        args.no_token_positional_embeddings = True
        args.adaptive_input = True
        args.encoder_normalize_before = False
        args.encoder_layers = args.num_layers
        args.encoder_embed_dim = args.hidden_dim
        args.encoder_attention_heads = args.num_heads
        args.attention_dropout = args.dropout
        args.encoder_ffn_embed_dim = args.hidden_dim
        self.args = args

        super().__init__(args, None, FeatureProjection(args))
        self.spatial_encoding = nn.Linear(5, args.hidden_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, image_features, feature_locations):
        # embed tokens and positions
        # print(image_features, image_features.shape)
        x = self.embed_scale * self.embed_tokens(image_features)

        # print(feature_locations.shape)

        if self.spatial_encoding is not None:
            x += self.spatial_encoding(feature_locations)

        x = self.dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        lengths = torch.zeros((image_features.shape[0])).fill_(
            image_features.shape[1]).to(self.args.device).long()
        encoder_padding_mask = create_padding_mask(image_features, lengths)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return x, encoder_padding_mask


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden = [batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # hidden = [batch size, src len, hid dim]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.output_dim = vocab_size
        self.attention = Attention(args.hidden_dim)

        self.embedding = nn.Embedding(vocab_size, args.hidden_dim)

        self.rnn = nn.GRU(args.hidden_dim + args.hidden_dim, args.hidden_dim, batch_first=True)

        self.fc_out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(args.hidden_dim + args.hidden_dim + args.hidden_dim, vocab_size))

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input, hidden, encoder_outputs):

        # input = [batch size]
        # hidden = [batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]

        input = input.unsqueeze(1)

        # input = [batch size, 1]

        embedded = self.dropout(self.embedding(input))

        # embedded = [batch size, 1, hidden dim]

        a = self.attention(hidden, encoder_outputs)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, hid dim]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, hid dim + hid dim]

        hidden = hidden.unsqueeze(0)
        output, hidden = self.rnn(rnn_input, hidden)

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(
            torch.cat((output, weighted, embedded), dim=-1))

        # prediction = [batch size, output dim]

        return prediction.squeeze(1), hidden.squeeze(0)
