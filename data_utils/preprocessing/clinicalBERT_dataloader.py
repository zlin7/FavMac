# https://github.com/kexinhuang12345/clinicalBERT/blob/master/run_readmission.py

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import csv
import logging
import os

import ipdb
import matplotlib as mpl
import tqdm

mpl.use('Agg')

import pandas as pd
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.ERROR)
logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, input_ids, input_mask, segment_ids, label_id, guid=None):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id
		#NOTE: Add guid=hadm_id-seq for future purposes
		self.guid = guid

class DataProcessor(object):
	"""Base class for data converters for sequence classification data sets."""

	def get_train_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the train set."""
		raise NotImplementedError()

	def get_dev_examples(self, data_dir):
		"""Gets a collection of `InputExample`s for the dev set."""
		raise NotImplementedError()

	def get_labels(self):
		"""Gets the list of labels for this data set."""
		raise NotImplementedError()

	@classmethod
	def _read_tsv(cls, input_file, quotechar=None):
		"""Reads a tab separated value file."""
		with open(input_file, "r") as f:
			reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
			lines = []
			for line in reader:
				lines.append(line)
			return lines

	@classmethod
	def _read_csv(cls, input_file):
		"""Reads a comma separated value file."""
		file=pd.read_csv(input_file)
		#lines=zip(file.ID,file.TEXT,file.Label)
		lines = zip(file.HADM_ID, file.TEXT, file.Seq)
		return lines

class readmissionProcessor(DataProcessor):
	def get_train_examples(self, data_dir):
		logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.csv")))
		return self._create_examples(
			self._read_csv(os.path.join(data_dir, "train.csv")), "train")

	def get_dev_examples(self, data_dir):
		return self._create_examples(
			self._read_csv(os.path.join(data_dir, "val.csv")), "val")

	def get_test_examples(self, data_dir):
		return self._create_examples(
			self._read_csv(os.path.join(data_dir, "test.csv")), "test")

	def get_examples(self, fpath):
		return self._create_examples(self._read_csv(fpath))

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines, set_type=None):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, line) in enumerate(lines):
			#TODO: Add Seq number and ID. Label is unimportant in our task.
			#guid = "%s-%s" % (set_type, i)
			guid = f"{line[0]}-{line[2]}"
			text_a = line[1]
			#label = str(int(line[2]))
			label = str(int(line[2]) % 2)
			examples.append(
				InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
		return examples



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i

	features = []
	for (ex_index, example) in tqdm.tqdm(enumerate(examples), total=len(examples)):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)

		if tokens_b:
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[0:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0     0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = []
		segment_ids = []
		tokens.append("[CLS]")
		segment_ids.append(0)
		for token in tokens_a:
			tokens.append(token)
			segment_ids.append(0)
		tokens.append("[SEP]")
		segment_ids.append(0)

		if tokens_b:
			for token in tokens_b:
				tokens.append(token)
				segment_ids.append(1)
			tokens.append("[SEP]")
			segment_ids.append(1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
			segment_ids.append(0)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		#print (example.label)
		label_id = label_map[example.label]
		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id,
							  guid=example.guid))
	return features