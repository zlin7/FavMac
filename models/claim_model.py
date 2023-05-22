from importlib import reload
from typing import Tuple, Optional

import ipdb, tqdm
import torch
import torch.nn as nn

import models.transformer as transformer; reload(transformer)
 
def mask_to_weight(msk, is_padding_msk=True):
    w = msk.float()
    if is_padding_msk: w = 1 - w
    w = w / w.sum(-1, keepdim=True).clip(1e-3)
    return w

def masked_mean(embeds, msk, is_padding_msk=True):
    assert len(embeds.shape) == len(msk.shape) + 1
    w = mask_to_weight(msk, is_padding_msk)
    embeds = embeds * (w.unsqueeze(-1))
    return embeds.sum(-2)

class ClaimDiagEmbedder(torch.nn.Module):
    def __init__(self, num_diags=887, embedding_dim=64, 
                 code_attn=True, ord_attn=True, 
                 visit_attn=True, max_visit_len=50) -> None:
        super().__init__()
        self.embedder = torch.nn.Embedding(num_diags, embedding_dim, padding_idx=0)

        self.code_attn, self.visit_attn, self.ord_attn = None, None, None

        if code_attn:
            self.code_attn = torch.nn.TransformerEncoderLayer(embedding_dim, 8, batch_first=True)
        if visit_attn:
            self.visit_attn = transformer.PositionalTransformerEncoder(max_visit_len, embedding_dim, 4, batch_first=True)
        if ord_attn:
            self.ord_attn = transformer.PositionalTransformerEncoder(13, embedding_dim, 4, batch_first=True)
        self.embedding_dim = embedding_dim
        self.max_visit_len = max_visit_len

    def agg_codes(self, x, pad_msk):
        # [batch, visit, seqord, codes]
        batch, nvisit, nseqord, ncodes = x.shape
        # assume the tail is padded
        ncodes = ncodes - pad_msk.reshape(-1, ncodes).sum(1).min().item()
        ncodes = max(ncodes, 1)
        x, pad_msk = x[..., :ncodes], pad_msk[..., :ncodes]
        x = self.embedder(x)
        if self.code_attn is not None:
            x_bc = x.reshape(-1, ncodes, self.embedding_dim)
            msk_bc = pad_msk.reshape(-1, ncodes)
            # msk_nonempty_codes = msk_bc.sum(1) < ncodes
            msk_nonempty_b = msk_bc.logical_not().any(-1)
            nonempty_out_bc = self.code_attn(x_bc[msk_nonempty_b], src_key_padding_mask=msk_bc[msk_nonempty_b])
            out_b = torch.zeros_like(x_bc[:, 0, :])
            out_b[msk_nonempty_b] = masked_mean(nonempty_out_bc, msk_bc[msk_nonempty_b])
            out = out_b.reshape(batch, nvisit, nseqord, self.embedding_dim)
        else:
            out = masked_mean(x, pad_msk)
        return out

    def agg_seqord(self, x, pad_msk):
        #NOTE: Add positional embeddings
        batch, nvisit, nseqord, embedding_dim = x.shape
        if nseqord == 1: return x.squeeze(2)
        if self.ord_attn is not None:
            x_bo = x.reshape(-1, nseqord, embedding_dim)
            msk_bo = pad_msk.reshape(-1, nseqord)
            msk_nonempty_b = msk_bo.logical_not().any(-1)
            nonempty_out_bo = self.ord_attn(x_bo[msk_nonempty_b], src_key_padding_mask=msk_bo[msk_nonempty_b])
            out_b = torch.zeros_like(x_bo[:, 0, :])
            out_b[msk_nonempty_b] = masked_mean(nonempty_out_bo, msk_bo[msk_nonempty_b])
            out = out_b.reshape(batch, nvisit, self.embedding_dim)
        else:
            out = masked_mean(x, pad_msk)
        return out

        
    def agg_visit(self, x, pad_msk):
        #NOTE: Add positional embeddings
        batch, nvisit, embedding_dim = x.shape
        if nvisit == 1: return x.squeeze(1)
        nvisit = nvisit - pad_msk.reshape(-1, nvisit).sum(1).min().item()
        nvisit = max(nvisit, 1)
        x, pad_msk = x[:, :nvisit,:], pad_msk[:, :nvisit]
        if self.visit_attn is not None:
            msk_nonempty_b = pad_msk.logical_not().any(-1)
            nonempty_out_bv= self.visit_attn(x[msk_nonempty_b], src_key_padding_mask=pad_msk[msk_nonempty_b])
            out = torch.zeros_like(x[:, 0, :])
            out[msk_nonempty_b] = masked_mean(nonempty_out_bv, pad_msk[msk_nonempty_b])
        else:
            out = masked_mean(x, pad_msk)
        return out

    def forward(self, diags):
        assert len(diags.shape) == 4,f"Invalid input of shape {diags.shape}"
        diags = diags[:, :self.max_visit_len]
        pad_msk = diags == 0
        # x: [batch, visit, seqord, codes] -> [batch, visit, seqord, embed]
        x = self.agg_codes(diags, pad_msk=pad_msk)

        # x: [batch, visit, seqord, embed] -> [batch, visit, embed]
        pad_msk = pad_msk.all(-1)
        x = self.agg_seqord(x, pad_msk=pad_msk)

        # x: [batch, visit, embed] -> [batch, embed]
        pad_msk = pad_msk.all(-1)
        # assert not pad_msk.all(-1).any(), "Shouldn't have empty"
        x = self.agg_visit(x, pad_msk=pad_msk)
        return x

    @classmethod
    def test(cls):
        import numpy as np
        embed_model = cls(code_attn=False, visit_attn=True, seqorder_attn=True)
        diags = np.zeros(shape=(2, 3, 4, 5)) #(pat, visit, seqord, codes)
        diags[0,0,0,0] = 32
        diags[0,0,0,1] = 1
        diags[0,0,1,0] = 25
        diags[1,0,0,0] = 12
        diags[1,1,0,0] = 13
        diags[1,2,0,0] = 58
        embed = embed_model.forward(torch.tensor(diags).int())

        
class ClaimModel(torch.nn.Module):
    def __init__(self, nclass, 
                 num_diags=887, diag_embed_dim=128, hidden_dim=128,
                 code_attn=True, ord_attn=True, visit_attn=True,
                  **kwargs) -> None:
        super().__init__()
        self.embedders = torch.nn.ModuleDict({
            'pat_sex': torch.nn.Embedding(2, 8),
            'pat_region': torch.nn.Embedding(5, 8),
            'diags': ClaimDiagEmbedder(num_diags, diag_embed_dim, 
                code_attn=code_attn, ord_attn=ord_attn, visit_attn=visit_attn)
        })
        cat_embedding_dim  = 1 + sum([_.embedding_dim for _ in self.embedders.values()])
        self.embedding_dim = hidden_dim
        self.fc1 = torch.nn.Linear(cat_embedding_dim, self.embedding_dim)
        self.fc2 = torch.nn.Linear(self.embedding_dim, nclass)
        

    def forward(self, input, embed_only=False, **kwargs):
        embeds = []
        for k in ['diags', 'pat_sex', 'pat_region']:
            embeds.append(self.embedders[k](input[k]))
        embeds.append(input['pat_age'].unsqueeze(-1).float())
        x = torch.concat(embeds, 1)

        x = torch.relu(self.fc1(x))
        if embed_only: return x
        return self.fc2(x)

class ClaimLogisticRegression(torch.nn.Module):
    def __init__(self, nclass, num_diags=887) -> None:
        super().__init__()
        input_dim = num_diags + 1 + 2 + 5 
        self.num_diags = num_diags
        self.fc = torch.nn.Linear(input_dim, nclass)
    def forward(self, input, embed_only=False, **kwargs):
        from torch.nn.functional import one_hot
        assert not embed_only
        x = input['diags']

        assert x.shape[1] == x.shape[2] == 1
        x = x.squeeze(1).squeeze(1)
        x = x[:, :max((x!=0).int().sum(1).max().item(), 1)]
        diag_embed = one_hot(x, num_classes=self.num_diags).max(1)[0]
        age_embed = input['pat_age'].unsqueeze(-1).float()
        reg_embed = one_hot(input['pat_region'], 5)
        sex_embed = one_hot(input['pat_sex'], 2)
        embed = torch.concat([diag_embed, age_embed, sex_embed, reg_embed], 1)
        return self.fc(embed)

if __name__ == '__main__':
    pass