from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import random


class Reader(object):
	def __init__(self, dataset_dir, listfile=None):
		self._dataset_dir = dataset_dir
		self._current_index = 0
		if listfile is None:
			listfile_path = os.path.join(dataset_dir, "listfile.csv")
		else:
			listfile_path = listfile
		with open(listfile_path, "r") as lfile:
			self._data = lfile.readlines()
		self._listfile_header = self._data[0]
		self._data = self._data[1:]

	def get_number_of_examples(self):
		return len(self._data)

	def random_shuffle(self, seed=None):
		if seed is not None:
			random.seed(seed)
		random.shuffle(self._data)

	def read_example(self, index):
		raise NotImplementedError()

	def read_next(self):
		to_read_index = self._current_index
		self._current_index += 1
		if self._current_index == self.get_number_of_examples():
			self._current_index = 0
		return self.read_example(to_read_index)


class PhenotypingReader(Reader):
	def __init__(self, dataset_dir, listfile=None):
		""" Reader for phenotype classification task.

		:param dataset_dir: Directory where timeseries files are stored.
		:param listfile:    Path to a listfile. If this parameter is left `None` then
							`dataset_dir/listfile.csv` will be used.
		"""
		Reader.__init__(self, dataset_dir, listfile)
		self._data = [line.split(',') for line in self._data]
		self._data = [(mas[0], float(mas[1]), list(map(int, mas[2:]))) for mas in self._data]

	def _read_timeseries(self, ts_filename):
		ret = []
		with open(os.path.join(self._dataset_dir, ts_filename), "r") as tsfile:
			header = tsfile.readline().strip().split(',')
			assert header[0] == "Hours"
			for line in tsfile:
				mas = line.strip().split(',')
				ret.append(np.array(mas))
		return (np.stack(ret), header)

	def read_example(self, index):
		""" Reads the example with given index.

		:param index: Index of the line of the listfile to read (counting starts from 0).
		:return: Dictionary with the following keys:
			X : np.array
				2D array containing all events. Each row corresponds to a moment.
				First column is the time and other columns correspond to different
				variables.
			t : float
				Length of the data in hours. Note, in general, it is not equal to the
				timestamp of last event.
			y : array of ints
				Phenotype labels.
			header : array of strings
				Names of the columns. The ordering of the columns is always the same.
			name: Name of the sample.
		"""
		if index < 0 or index >= len(self._data):
			raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")

		name = self._data[index][0]
		t = self._data[index][1]
		y = self._data[index][2]
		(X, header) = self._read_timeseries(name)

		return {"X": X,
				"t": t,
				"y": y,
				"header": header,
				"name": name}

