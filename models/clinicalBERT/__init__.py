import os

import pandas as pd

import _settings
from models.clinicalBERT.modeling_readmission import (
    BertForSequenceClassification, BertModel)

OUTPUT_PATH = os.path.join(_settings.WORKSPACE, 'pretrained')
if not os.path.isdir(OUTPUT_PATH):
	os.makedirs(OUTPUT_PATH)
PRETRAINED_PATH = os.path.join(OUTPUT_PATH, 'ClinicalBERT_checkpoint', 'ClinicalBERT_pretraining_pytorch_checkpoint')

def load_pretrained_BERT_classifier(num_labels=7):
	#return BertForSequenceClassification.from_pretrained(PRETRAINED_PATH, num_labels=num_labels)
	return BertModel.from_pretrained(PRETRAINED_PATH)

