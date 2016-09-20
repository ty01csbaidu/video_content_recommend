#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_matrix.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 下午4:44


import logging
import heapq
import numpy as np
from similarity import DescSimilarity, TagSimilarity, TitleSimilarity, StarSimilarity
from structural_similarity import LinearStructuralSimilarity
from scipy.spatial.distance import cosine
import time


logger = logging.getLogger('simialrities.similarity_matrix')


class ScorePair(object):
	"""
	similarity score pair
	"""
	def __init__(self, idx=0, s=0):
		self.idx = idx
		self.s = s

	def __lt__(self, other):
		return self.s < other.s


class SimilarityMatrix(object):

	def __init__(self, word2vec_model, topN):
		self.word2vec_model = word2vec_model
		self.topN = topN
		self.doc_vector = {}
		self.s_matrix = {}

	def run(self, old_similarity_file, old_doc_vector_file, old_videos, matrix_out, doc_vector_out, videos):
		self.load_similarity(old_similarity_file)
		self.load_doc_vector(old_doc_vector_file)
		self.compute(old_videos, videos)
		self.save_simialrity(matrix_out)
		self.save_doc_vector(doc_vector_out)

	def load_similarity(self, old_similarity_file):
		"""
		load old similarity matrix
		format: 2	6586:1.51655394401,293641:1.47868037277,293916:1.45655895399,293617:1.43322061068,293904:1.35915309858
		:param old_similarity_file:
		:return:
		"""
		with open(old_similarity_file) as old_sim_f:
			for line in old_sim_f:
				fields = line.strip('\n').strip('\t').split('\t')
				if len(fields) == 2:
					s_vid = fields[0]
					score_str = fields[1]
					score_list = score_str.strip(' ').strip(',').split(',')
					if len(score_list) > 0:
						score_pair_list = []
						for item in score_list:
							d_vid, score = item.strip(' ').split(':')
							score = float(score)
							sp = ScorePair(d_vid, score)
							score_pair_list.append(sp)
						heapq.heapify(score_pair_list)
						self.s_matrix[s_vid] = score_pair_list

	def save_simialrity(self, matrix_out):
		with open(matrix_out, 'w') as m_o:
			for vid, top_similarity in self.s_matrix.items():
				ret_list = []
				for item in top_similarity:
					s_vid = item.idx
					score = item.s
					ret_list.append(str(s_vid) + ':' + str(score))
				m_o.write(str(vid) + '\t' + ','.join(ret_list) + '\n')

	def load_doc_vector(self, old_doc_vector_file):
		"""
		format: vid \t vector(,)
		:param old_doc_vector_file:
		:return:
		"""
		with open(old_doc_vector_file) as old_doc_vector_f:
			for line in old_doc_vector_f:
				fields = line.strip('\n').strip('\t').split('\t')
				if len(fields) == 2:
					vid = fields[0]
					v_list = fields[1]
					vector = np.array(list[v_list.strip(',').split(',')])
					self.doc_vector[vid] = vector

	def save_doc_vector(self, doc_vector_out):
		with open(doc_vector_out, 'w') as doc_vector_o:
			for vid, vector in self.doc_vector.items():
				ret = vid + '\t'
				v_list = [str(item) for item in vector]
				ret += ','.join(v_list)
				ret += '\n'
				doc_vector_o.write(ret)

	def compute_doc_vector(self, videos):
		"""
		:param videos:
		:return:
		"""
		desc_similarity = DescSimilarity()
		start_time = time.time()
		for i, video in enumerate(videos):
			self.doc_vector[video.vid] = (desc_similarity.document_vector(video.desc, self.word2vec_model))

		print("doc vector time: %s" % (time.time() - start_time))

	def compute(self, old_videos, videos):
		"""
		compute similarity matrix
		first compute similarity between old_video and videos
		then compute inner similarity of videos
		:param videos:
		:return:
		"""
		linear_simialarity = LinearStructuralSimilarity([1.0, 0.5, 0.5, 0.5])
		title_similarity = TitleSimilarity()
		tag_similarity = TagSimilarity()
		star_similarity = StarSimilarity()

		#compute doc vector for new vector
		self.compute_doc_vector(videos)
		#init s_matrix for new vector
		for i, new_video in enumerate(videos):
			matrix_row = []
			heapq.heapify(matrix_row)
			self.s_matrix[new_video.vid] = matrix_row

		#compute similarity between old video and videos
		for i, old_video in enumerate(old_videos):
			old_vid = old_video.vid
			if old_vid in self.doc_vector and old_vid in self.s_matrix:
				for j, new_video in enumerate(videos):
					new_vid = new_video.vid
					desc_score = 1 - cosine(self.doc_vector[old_vid], self.doc_vector[new_vid])
					title_score = title_similarity.cosin_compute(old_video.name, new_video.name)
					tag_score = tag_similarity.compute(old_video.tags, new_video.tags)
					star_score = star_similarity.compute(old_video.stars, new_video.stars)
					linear_simialarity.set_similarity([desc_score, title_score, tag_score, star_score])
					s = linear_simialarity.compute()
					sp = ScorePair(new_vid, s)
					if len(self.s_matrix[old_vid]) < self.topN:
						self.s_matrix[old_vid].append(sp)
					else:
						heapq.heappushpop(self.s_matrix[old_vid], sp)
					s_sp = ScorePair(old_vid, s)
					if len(self.s_matrix[new_vid]) < self.topN:
						self.s_matrix[new_vid].append(s_sp)
					else:
						heapq.heappushpop(self.s_matrix[new_vid], s_sp)

		#compute inner simialrity between videos
		for i, fir_video in enumerate(videos):
			print "compute similarity documents: " + str(i)
			fir_vid = fir_video.vid
			for j, sec_video in enumerate(videos):
				sec_vid = sec_video.vid
				if i < j:
					desc_score = 1 - cosine(self.doc_vector[fir_vid], self.doc_vector[sec_vid])
					title_score = title_similarity.cosin_compute(fir_video.name, sec_video.name)
					tag_score = tag_similarity.compute(fir_video.tags, sec_video.tags)
					star_score = star_similarity.compute(fir_video.stars, sec_video.stars)
					linear_simialarity.set_similarity([desc_score, title_score, tag_score, star_score])
					s = linear_simialarity.compute()
					sp = ScorePair(j, s)
					if len(self.s_matrix[fir_vid]) < self.topN:
						self.s_matrix[fir_vid].append(sp)
					else:
						heapq.heappushpop(self.s_matrix[fir_vid], sp)
					s_sp = ScorePair(i, s)
					if len(self.s_matrix[sec_vid]) < self.topN:
						self.s_matrix[sec_vid].append(s_sp)
					else:
						heapq.heappushpop(self.s_matrix[sec_vid], s_sp)

		for vid, matrix_row in self.s_matrix.items():
			self.s_matrix[vid] = heapq.nlargest(self.topN, matrix_row)







