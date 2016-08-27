#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_matrix.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 下午4:44


import logging
from similarity import DescSimilarity, TagSimilarity, TitleSimilarity, StarSimilarity
from structural_similarity import LinearStructuralSimilarity
from scipy.spatial.distance import cosine


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

		#transform documents to vector firstly, then compute score
		doc_vector = []
		for i, video in enumerate(videos):
			doc_vector.append(desc_similarity.document_vector(video.desc, self.word2vec_model))
		for i, fir_video in enumerate(videos):
			print "compute similarity documents: " + str(i)
			idx_dict[i] = fir_video.vid
			matrix_row = {}
			for j, sec_video in enumerate(videos):
				if i < j:
					#desc_score = desc_similarity.compute(fir_video.desc, sec_video.desc, self.word2vec_model)
					desc_score = cosine(doc_vector[i], doc_vector[j])
					#title_score = title_similarity.compute(fir_video.name, sec_video.name, self.word2vec_model)
					title_score = title_similarity.cosin_compute(fir_video.name, sec_video.name)
					tag_score = tag_similarity.compute(fir_video.tag, sec_video.tag)
					star_score = star_similarity.compute(fir_video.stars, sec_video.stars)
					linear_simialarity.set_similarity([desc_score, title_score, tag_score, star_score])
					s = linear_simialarity.compute()
					matrix_row[j] = s

				elif i > j:
					#symmetric
					matrix_row[j] = s_matrix[j][i]


			s_matrix.append(matrix_row)

		for i, matrix_row in enumerate(s_matrix):
			matrix_row = matrix_row.items()
			# topN
			matrix_row = sorted(matrix_row, key=lambda x:x[1], reverse=True)
			s_matrix[i] = matrix_row[:self.topN]

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






