#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_matrix.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 下午4:44


import logging
from src.models.video import Video, VideoIterable
from similarity import DescSimilarity, TagSimilarity, TitleSimilarity, StarSimilarity
from structural_similarity import LinearStructuralSimilarity


logger = logging.getLogger('simialrities.similarity_matrix')

class SimilarityMatrix(object):

	def __init__(self, matrix_out, word2vec_model):
		self.matrix_out = matrix_out
		self.word2vec_model = word2vec_model

	def compute(self, videos):
		"""
		compute similarity matrix
		:param videos:
		:return:
		"""
		linear_simialarity = LinearStructuralSimilarity([1.0, 0.5, 0.5, 0.5])
		desc_similarity = DescSimilarity()
		title_similarity = TitleSimilarity()
		tag_similarity = TagSimilarity()
		star_similarity = StarSimilarity()

		s_matrix = []
		for i, fir_video in enumerate(videos):
			matrix_row = []
			for j, sec_video in enumerate(videos):
				if not i == j:
					desc_similarity.compute(fir_video.desc, sec_video.desc, self.word2vec_model)
					title_similarity.compute(fir_video.title, sec_video.title, self.word2vec_model)
					tag_similarity.compute(fir_video.tag, sec_video.tag)
					star_similarity.compute(fir_video.star, sec_video.star)
					linear_simialarity.set_similarity([desc_similarity, title_similarity, tag_similarity, star_similarity])
					s = linear_simialarity.compute()
				else:
					s = 1.0
				matrix_row.append(s)
			s_matrix.append(matrix_row)



