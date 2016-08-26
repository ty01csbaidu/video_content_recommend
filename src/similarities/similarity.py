#!/usr/bin/env python
# -*- coding: utf-8 -*-
############################################################################
## 
## Copyright (c) 2013 hunantv.com, Inc. All Rights Reserved
## $Id: similarity.py,v 0.0 2016年08月17日 星期三 16时12分40秒  <> Exp $ 
## 
############################################################################
#
###
# # @file   similarity.py 
# # @author <tangye><<tangye@mgtv.com>>
# # @date   2016年08月17日 星期三 16时12分40秒  
# # @brief 
# #  
# ##

import logging
import itertools

import numpy as np
from scipy.spatial.distance import cosine

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import	CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger('similarities.similarity')


class Similarity(object):
	"""
	Base class for similarities
	"""

	def n_gram_similarity(self, vectorizer, x, y):
		"""
		n_gram similarity using sklearn
		:param vectorizer:
		:param x:
		:param y:
		:return:
		"""
		x_vector = vectorizer.transform(x)
		y_vector = vectorizer.transform(y)
		score = cosine_similarity(x_vector,y_vector)
		return score

	def set_cosin_similarity(self, x, y):
		"""
		s = intersection(x,y)/union(x,y)
		:param x:
		:param y:
		:return:
		"""
		if len(x) == 0 or len(y) == 0:
			return 0
		else:
			intersection = x & y
			union = x | y
			score = float(len(intersection)) / len(union)
			return score

	def compute(self, x, y, model=None):
		raise NotImplementedError("not implement a abstract method")


class DescSimilarity(Similarity):
	"""
	video description similarity using word2vec or docsim or lda
	"""

	def compute(self, x, y, word2vec_model):
		"""
		1. average over all words vector in a document, then compute cosin similarity
		2. using docsim directly
		3. using topic model for documents similarity
		:param x:
		:param y:
		:return:
		"""
		score = 0
		if x is not None and y is not None:
			x = [item for item in x if item in word2vec_model.vocab]
			y = [item for item in y if item in word2vec_model.vocab]
			if len(x) > 0 and len(y) > 0:
				#x_array = np.asarray([word2vec_model[item] for item in x])
				#y_array = np.asarray([word2vec_model[item] for item in y])
				#x_mean = np.mean(x_array, axis=0)
				#y_mean = np.mean(y_array, axis=0)
				#score = cosine(x_mean, y_mean)
				score = word2vec_model.n_similarity(x, y)
		return score


class TitleSimilarity(Similarity):
	"""
	video title similarity
	"""
	def compute(self, x, y, word2vec_model):
		"""
		1. average over all words vector in a document, then compute cosin similarity
		2. n-gram vector(or lsi)
		:param x:
		:param y:
		:param word2vec_model:
		:return:
		"""
		score = 0
		if x is not None and y is not None:
			x = [item for item in x if item in word2vec_model.vocab]
			y = [item for item in y if item in word2vec_model.vocab]
			if len(x) > 0 and len(y) > 0:
				#x_array = np.asarray([word2vec_model[item] for item in x])
				#y_array = np.asarray([word2vec_model[item] for item in y])
				#x_mean = np.mean(x_array, axis=0)
				#y_mean = np.mean(y_array, axis=0)
				#score = cosine(x_mean, y_mean)
				score = word2vec_model.n_similarity(x, y)
		return score


class TagSimilarity(Similarity):
	"""
	video tag similarity
	"""
	def __init__(self):
		self.filtered_tags = set(['剧情','偶像','全部剧集'])

	def compute(self, x, y):
		score = 0
		if x is not None and y is not None:
			x = x - self.filtered_tags
			y = y - self.filtered_tags
			score = self.set_cosin_similarity(x, y)
		return score


class StarSimilarity(Similarity):
	"""
	video stars similarity
	"""
	def __init__(self):
		self.filtered_stars = set(['暂无','无'])

	def compute(self, x, y):
		score = 0
		if x is not None and y is not None:
			x = x - self.filtered_stars
			y = y - self.filtered_stars
			score = self.set_cosin_similarity(x, y)
		return score

## vim: set ts=2 sw=2: #

