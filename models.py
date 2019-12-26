import torch
import torch.nn as nn
import torchvision
import torch.optim
import torch.utils.data
from torch.nn import init
import torch.nn.functional as F
import math
import utils

from typing import Tuple
from torch import Tensor

# CelebA (VAE)
# Input 64x64x3.
# Adam 1e-4
# Encoder Conv 32x4x4 (stride 2), 32x4x4 (stride 2), 64x4x4 (stride 2),
# 64x4x4 (stride 2), FC 256. ReLU activation.
# Latents 32
# Decoder Deconv reverse of encoder. ReLU activation. Gaussian.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BetaVAE(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            self._conv(3, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 3),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(kl_diverge, recon_loss)

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)


class DFCVAE(BetaVAE):

    def __init__(self, latent_size=100, beta=1):
        super(DFCVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.e1 = self._conv(3, 32)
        self.e2 = self._conv(32, 64)
        self.e3 = self._conv(64, 128)
        self.e4 = self._conv(128, 256)
        self.e5 = self._conv(256, 512)
        self.e6 = self._conv(512, 1024)
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_var = nn.Linear(4096, latent_size)

        # decoder
        self.d1 = self._upconv(1024, 512)
        self.d2 = self._upconv(512, 256)
        self.d3 = self._upconv(256, 128)
        self.d4 = self._upconv(128, 64)
        self.d5 = self._upconv(64, 32)
        self.d6 = self._upconv(32, 3)
        self.fc_z = nn.Linear(latent_size, 4096)

    def encode(self, x):
        x = F.leaky_relu(self.e1(x))
        x = F.leaky_relu(self.e2(x))
        x = F.leaky_relu(self.e3(x))
        x = F.leaky_relu(self.e4(x))
        x = F.leaky_relu(self.e5(x))
        x = F.leaky_relu(self.e6(x))
        x = x.view(-1, 4096)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 1024, 4, 4)
        z = F.leaky_relu(self.d1(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d2(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d3(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d4(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d5(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d6(F.interpolate(z, scale_factor=2)))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )

class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Encoder, self).__init__()
        #resnet = torchvision.models.resnet101(pretrained = True)
        resnet = torchvision.models.resnet101(pretrained=True)
        all_modules = list(resnet.children())
        # Remove the last FC layer used for classification and the average pooling layer
        modules = all_modules[:-2]
        # Initialize the modified resnet as the class variable
        self.resnet = nn.Sequential(*modules)
        self.avgpool = nn.AvgPool2d(7)
        self.fine_tune()    # To fine-tune the CNN, self.fine_tune(status = True)

    def forward(self, images):
        """
        The forward propagation function
        input: resized image of shape (batch_size,3,224,224)
        """
        # Run the image through the ResNet
        encoded_image = self.resnet(images)         # (batch_size,2048,7,7)
        batch_size = encoded_image.shape[0]
        features = encoded_image.shape[1]
        num_pixels = encoded_image.shape[2] * encoded_image.shape[3]
        # Get the global features of the image
        global_features = self.avgpool(encoded_image).view(
            batch_size, -1)   # (batch_size, 2048)
        enc_image = encoded_image.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        # (batch_size,num_pixels,2048)
        enc_image = enc_image.view(batch_size, num_pixels, features)
        return enc_image, global_features

    def fine_tune(self, status=False):

        if not status:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            # 1 layer only. len(list(resnet.children())) = 8
            for module in list(self.resnet.children())[7:]:
                for param in module.parameters():
                    param.requires_grad = True


class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdaptiveLSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.x_gate = nn.Linear(input_size, hidden_size)
        self.h_gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp, states):
        h_old, c_old = states
        ht, ct = self.lstm_cell(inp, (h_old, c_old))
        sen_gate = F.sigmoid(self.x_gate(inp) + self.h_gate(h_old))
        st = sen_gate * F.tanh(ct)
        return ht, ct, st


class AdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, att_dim):
        super(AdaptiveAttention, self).__init__()
        self.sen_affine = nn.Linear(hidden_size, hidden_size)
        self.sen_att = nn.Linear(hidden_size, att_dim)
        self.h_affine = nn.Linear(hidden_size, hidden_size)
        self.h_att = nn.Linear(hidden_size, att_dim)
        self.v_att = nn.Linear(hidden_size, att_dim)
        self.alphas = nn.Linear(att_dim, 1)
        self.context_hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, spatial_image, decoder_out, st):
        """
        spatial_image: the spatial image of size (batch_size,num_pixels,hidden_size)
        decoder_out: the decoder hidden state of shape (batch_size, hidden_size)
        st: visual sentinal returned by the Sentinal class, of shape: (batch_size, hidden_size)
        """
        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        num_pixels = spatial_image.shape[1]
        # (batch_size,num_pixels,att_dim)
        visual_attn = self.v_att(spatial_image)
        # (batch_size,hidden_size)
        sentinel_affine = F.relu(self.sen_affine(st))
        sentinel_attn = self.sen_att(
            sentinel_affine)     # (batch_size,att_dim)

        # (batch_size,hidden_size)
        hidden_affine = F.tanh(self.h_affine(decoder_out))
        # (batch_size,att_dim)
        hidden_attn = self.h_att(hidden_affine)

        hidden_resized = hidden_attn.unsqueeze(1).expand(
            hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1))

        # (batch_size, num_pixels+1, hidden_size)
        concat_features = torch.cat(
            [spatial_image, sentinel_affine.unsqueeze(1)], dim=1)
        # (batch_size, num_pixels+1, att_dim)
        attended_features = torch.cat(
            [visual_attn, sentinel_attn.unsqueeze(1)], dim=1)

        # (batch_size, num_pixels+1, att_dim)
        attention = F.tanh(attended_features + hidden_resized)

        # (batch_size, num_pixels+1)
        alpha = self.alphas(attention).squeeze(2)
        # (batch_size, num_pixels+1)
        att_weights = F.softmax(alpha, dim=1)

        context = (concat_features * att_weights.unsqueeze(2)
                   ).sum(dim=1)       # (batch_size, hidden_size)
        # (batch_size, 1)
        beta_value = att_weights[:, -1].unsqueeze(1)

        out_l = F.tanh(self.context_hidden(context + hidden_affine))

        return out_l, att_weights, beta_value


class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, vocab_size, att_dim, embed_size, encoded_dim):
        super(DecoderWithAttention, self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.encoded_to_hidden = nn.Linear(encoded_dim, hidden_size)
        self.global_features = nn.Linear(encoded_dim, embed_size)
        self.LSTM = AdaptiveLSTMCell(embed_size * 2, hidden_size)
        self.adaptive_attention = AdaptiveAttention(hidden_size, att_dim)
        # input to the LSTMCell should be of shape (batch, input_size). Remember we are concatenating the word with
        # the global image features, therefore out input features should be embed_size * 2
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=0.5)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, enc_image):
        h = torch.zeros(enc_image.shape[0], 512).to(device)
        c = torch.zeros(enc_image.shape[0], 512).to(device)
        return h, c

    def forward(self, enc_image, global_features, encoded_captions, caption_lengths):
        """
        enc_image: the encoded images from the encoder, of shape (batch_size, num_pixels, 2048)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, 2048)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        spatial_image = F.relu(self.encoded_to_hidden(
            enc_image))  # (batch_size,num_pixels,hidden_size)
        global_image = F.relu(self.global_features(
            global_features))      # (batch_size,embed_size)
        batch_size = spatial_image.shape[0]
        num_pixels = spatial_image.shape[1]
        # Sort input data by decreasing lengths
        # caption_lenghts will contain the sorted lengths, and sort_ind contains the sorted elements indices
        caption_lengths, sort_ind = caption_lengths.squeeze(
            1).sort(dim=0, descending=True)
        # The sort_ind contains elements of the batch index of the tensor encoder_out. For example, if sort_ind is [3,2,0],
        # then that means the descending order starts with batch number 3,then batch number 2, and finally batch number 0.
        # (batch_size,num_pixels,hidden_size) with sorted batches
        spatial_image = spatial_image[sort_ind]
        # (batch_size, embed_size) with sorted batches
        global_image = global_image[sort_ind]
        # (batch_size, max_caption_length) with sorted batches
        encoded_captions = encoded_captions[sort_ind]
        # (batch_size, num_pixels, 2048)
        enc_image = enc_image[sort_ind]

        # Embedding. Each batch contains a caption. All batches have the same number of rows (words), since we previously
        # padded the ones shorter than max_caption_length, as well as the same number of columns (embed_dim)
        # (batch_size, max_caption_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

        # Initialize the LSTM state
        # (batch_size, hidden_size)
        h, c = self.init_hidden_state(enc_image)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(
            decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(
            decode_lengths), num_pixels+1).to(device)
        betas = torch.zeros(batch_size, max(decode_lengths), 1).to(device)

        # Concatenate the embeddings and global image features for input to LSTM
        global_image = global_image.unsqueeze(1).expand_as(embeddings)
        # (batch_size, max_caption_length, embed_dim * 2)
        inputs = torch.cat((embeddings, global_image), dim=2)

        # Start decoding
        for timestep in range(max(decode_lengths)):
            # Create a Packed Padded Sequence manually, to process only the effective batch size N_t at that timestep. Note
            # that we cannot use the pack_padded_seq provided by torch.util because we are using an LSTMCell, and not an LSTM
            batch_size_t = sum([l > timestep for l in decode_lengths])
            # (batch_size_t, embed_dim * 2)
            current_input = inputs[:batch_size_t, timestep, :]
            # (batch_size_t, hidden_size)
            h, c, st = self.LSTM(
                current_input, (h[:batch_size_t], c[:batch_size_t]))
            # Run the adaptive attention model
            out_l, alpha_t, beta_t = self.adaptive_attention(
                spatial_image[:batch_size_t], h, st)
            # Compute the probability over the vocabulary
            # (batch_size, vocab_size)
            pt = self.fc(self.dropout(out_l))
            predictions[:batch_size_t, timestep, :] = pt
            alphas[:batch_size_t, timestep, :] = alpha_t
            betas[:batch_size_t, timestep, :] = beta_t
        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind

