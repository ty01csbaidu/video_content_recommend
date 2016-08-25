#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_matrix.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 下午4:44


import logging
from similarity import DescSimilarity, TagSimilarity, TitleSimilarity, StarSimilarity
from structural_similarity import LinearStructuralSimilarity


logger = logging.getLogger('simialrities.similarity_matrix')

class SimilarityMatrix(object):

	def __init__(self, matrix_out, word2vec_model, topN):
		self.matrix_out = matrix_out
		self.word2vec_model = word2vec_model
		self.topN = topN

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
		idx_dict = {}
		for i, fir_video in enumerate(videos):
			idx_dict[i] = fir_video.vid
			matrix_row = []
			for j, sec_video in enumerate(videos):
				if not i == j:
					desc_similarity.compute(fir_video.desc, sec_video.desc, self.word2vec_model)
					title_similarity.compute(fir_video.title, sec_video.title, self.word2vec_model)
					tag_similarity.compute(fir_video.tag, sec_video.tag)
					star_similarity.compute(fir_video.star, sec_video.star)
					linear_simialarity.set_similarity([desc_similarity, title_similarity, tag_similarity, star_similarity])
					s = linear_simialarity.compute()
					matrix_row.append((j, s))
			# topN
			matrix_row = sorted(matrix_row, key=lambda x:x[1], reverse=True)
			s_matrix.append(matrix_row[:self.topN])

		return (s_matrix, idx_dict)

	def save(self, s_matrix, idx_dict):
		with open(self.matrix_out, 'w') as m_o:
			for id, top_similarity in enumerate(s_matrix):
				if id in idx_dict:
					vid = idx_dict[id]
					ret_list = []
					for item in top_similarity:
						s_id = item[0]
						score = item[1]
						if s_id in idx_dict:
							s_vid = idx_dict[s_id]
							ret_list.append(str(s_vid) + ':' + str(score))
					m_o.write(str(vid) + '\t' + ','.join(ret_list) + '\n')






