import math

import torch
import torch.nn as nn

import hparams as hp
from utils import training_utils


class TextEncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            dropout,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.conv1d(x)
        z = self.batch_norm(z)
        z = self.relu(z)
        z = self.dropout(z)
        return z


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, bs, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0).unsqueeze(0).repeat(bs, 1, 1)
        self.pe = pe

    def forward(self, lens):
        mask = training_utils.get_mask_from_lens(lens)
        pos_enc = self.pe[:, :, :lens.max().type(torch.LongTensor)].cuda() * mask.type(torch.FloatTensor).cuda()
        return pos_enc.permute(2, 0, 1)


class AlignmentEncoder(nn.Module):
    def __init__(self, embedding_dim, bidir_enc):
        super().__init__()
        self.lstm_hidden = embedding_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            bidirectional=bidir_enc,
        )

    def forward(self, z):
        z, _ = self.lstm(z)
        z = (z[:, :, :self.lstm_hidden] + z[:, :, self.lstm_hidden:]).permute(1, 2, 0)
        z = z.permute(2, 0, 1)
        return z


class LengthPredictor(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, dropout):
        super().__init__()
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv1d_1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.conv1d_2 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        z = self.conv1d_1(x)
        z = self.layer_norm1(z.permute(0, 2, 1))
        z = self.dropout1(z)
        z = self.conv1d_2(z.permute(0, 2, 1))
        z = self.layer_norm2(z.permute(0, 2, 1))
        z = self.dropout2(z)
        z = self.proj(z.permute(0, 2, 1))
        z = self.relu(z)
        z = torch.sum(z, dim=(1, 2))
        return z
