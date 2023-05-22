import math

import ipdb
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.batch_first = batch_first
		if batch_first: pe = pe.permute(1,0,2)
		self.register_buffer('pe', pe)

	def forward(self, x: Tensor) -> Tensor:
		"""
		Args:
			x: Tensor, shape [seq_len, batch_size, embedding_dim]
		"""
		if self.batch_first:
			x = x + self.pe[:, :x.size(1)]
		else:
			x = x + self.pe[:x.size(0)]
		if self.dropout is not None:
			x = self.dropout(x)
		return x



class PositionalTransformerEncoder(torch.nn.Module):
	def __init__(self, max_seq_len, embed_size, num_heads=4, batch_first=True, nlayers=1, **kwargs) -> None:
		super().__init__()

		encoder_layers = torch.nn.TransformerEncoderLayer(embed_size, num_heads, batch_first=batch_first, **kwargs)
		self.pos_encoder = PositionalEncoding(embed_size, max_len=max_seq_len, batch_first=batch_first)
		self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
		self.batch_first = batch_first

	def forward(self, x: Tensor, **kwargs) -> Tensor:
		x = self.pos_encoder(x)
		return self.transformer_encoder(x, **kwargs)