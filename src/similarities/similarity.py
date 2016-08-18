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

	def compute(self, x, y):
		raise NotImplementedError("not implement a abstract method")


class DescSimilarity(Similarity):
	"""
	video description similarity using word2vec or docsim or lda
	"""
	def load_model(self, model_out):
		self.model = Word2vec.load_word2vec_format(model_out, binary=True)

	def compute(self, x, y):
		"""
		1. average over all words vector in a document, then compute cosin similarity
		2. using docsim directly
		3. using topic model for documents similarity
		:param x:
		:param y:
		:return:
		"""
		score = self.model.n_similarity(x, y)



class TitleSimilarity(Similarity):
	"""
	video title similarity
	"""


class TagSimilarity(Similarity):
	"""
	video tag similarity
	"""

class StarSimilarity(Similarity):
	"""
	video stars similarity
	"""
## vim: set ts=2 sw=2: #

