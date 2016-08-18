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

import numpy

from gensim.models import Word2vec

logger = logging.getLogger('similarities.similarity')


class Similarity(object):
	"""
	Base class for similarities
	"""

	def n_gram_similarity(self, x, y, n=1):
		"""
		n_gram vector similarity for documents
		:param x:
		:param y:
		:return:
		"""


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
		score = word2vec_model.n_similarity(x, y)
		return score


class TagSimilarity(Similarity):
	"""
	video tag similarity
	"""
	def __init__(self):
		self.filtered_tags = set(['剧情','偶像','全部剧集'])

	def compute(self, x, y):
		score = self.n_gram_simialrity(x, y, 1)
		return score

class StarSimilarity(Similarity):
	"""
	video stars similarity
	"""
	def __init__(self):
		self.filtered_stars = set(['暂无','无'])

	def compute(self, x, y):
		score = self.n_gram_similarity(x, y, 1)
		return score
## vim: set ts=2 sw=2: #

