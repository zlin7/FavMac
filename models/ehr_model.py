import os.path
from importlib import reload

import ipdb
import torch
import tqdm

import models.basic_models


class MIMIC_MLP(models.basic_models.MLP):
    def __init__(self, nclass, input_dim=589) -> None:
        super().__init__(input_dim=input_dim, hdims=[256, 256], nclass=nclass)
        assert nclass == 7

class ClinicalBERTEncoder(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        from .clinicalBERT import load_pretrained_BERT_classifier
        self.bert = load_pretrained_BERT_classifier()
        self.embedding_size = self.bert.classifier.in_features
        setattr(self.bert, 'classifier',  torch.nn.Identity())
        # embedding size = 768

    def forward(self, input_ids, segment_ids, input_mask, label_ids=None):
        embeds = self.bert(input_ids, segment_ids, input_mask)
        return embeds


class EHRModel(torch.nn.Module):
    def __init__(self, nclass, use_note=False, **kwargs) -> None:
        super().__init__()

        embed_size = 128 + nclass
        #self.event_encoder = models.basic_models.MLP(input_dim=589, hdims=[128, 128], nclass=1)
        self.event_encoder = models.basic_models.MLP(input_dim=609, hdims=[128, 128], nclass=1)
        self.event_encoder.readout = torch.nn.Identity()
        if use_note:
            from .clinicalBERT import load_pretrained_BERT_classifier
            self.bert_encoder = load_pretrained_BERT_classifier()
            embed_size += 768
        self.use_note = use_note

        self.fc = torch.nn.Linear(embed_size, nclass)
        #self.fc = torch.nn.Linear(128, nclass)


    def get_readout_layer(self):
        return self.fc

    def get_embed_size(self):
        return self.fc.in_features

    def aggregate_bert_embeddings(self, bert_embeds, num_blocks, method='mean'):

        if method == 'mean':
            ret = []
            start = 0
            for i, l in enumerate(num_blocks):
                ret.append(torch.mean(bert_embeds[start:start+l], 0, keepdim=True))
                start += l
        else:
            raise NotImplementedError()
        return torch.concat(ret), None

    def embed_notes(self, input):
        num_blocks = input['num_blocks'].detach().cpu().numpy()
        # compute the embedding for the language model.
        input_ids = torch.concat([input['input_ids'][i, :l] for i, l in enumerate(num_blocks)])
        segment_ids = torch.concat([input['segment_ids'][i, :l] for i, l in enumerate(num_blocks)])
        input_mask = torch.concat([input['input_mask'][i, :l] for i, l in enumerate(num_blocks)])
        _, bert_embeds = self.bert_encoder(input_ids, segment_ids, input_mask, output_all_encoded_layers=True)
        bert_embeds, bert_preds = self.aggregate_bert_embeddings(bert_embeds, num_blocks)
        return bert_embeds


    def forward(self, input, embed_only=False, **kwargs):
        event = input['event']
        event_embeds =self.event_encoder(event)
        partialHCC = input['partialHCC']
        if self.use_note:
            bert_embeds = self.embed_notes(input)
            embeds = torch.concat([bert_embeds, event_embeds, partialHCC], axis=1)
        else:
            embeds = torch.concat([event_embeds, partialHCC], axis=1)
        if embed_only: return embeds
        return self.fc(embeds)
