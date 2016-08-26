#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: structural_similarity.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/19 上午11:00


import logging
import numpy as np


logger = logging.getLogger('similarities.structural_similarity')

class StructuralSimilarity(object):
	"""
	video structual similarity
	"""

	def compute(self):
		raise NotImplementedError("not implement a abstract method")


class LinearStructuralSimilarity(StructuralSimilarity):
	"""
	linear weight of unit similarities
	"""
	def __init__(self, weights):
		self.weights = weights

	def set_similarity(self, unit_similarities):
		self.unit_similarities = unit_similarities

	def set_linear_weight(self, weights):
		self.weights = weights

	def compute(self):
		assert(len(self.weights) == len(self.unit_similarities))
		#print self.unit_similarities
		#print self.weights
		w = np.asarray(self.weights, dtype=float)
		s = np.asarray(self.unit_similarities, dtype=float)
		score = np.dot(w, s)
		return score




