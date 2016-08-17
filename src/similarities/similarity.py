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
	def __init__(self, num_features):
		self.num_features = num_features

	def compute(self, x, y):
		raise NotImplementedError("not implement a abstract method")


class DescSimilarity(Similarity):
	"""
	video description similarity using word2vec or lda
	"""

## vim: set ts=2 sw=2: #

