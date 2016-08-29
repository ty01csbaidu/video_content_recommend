#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_matrix.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 下午4:44


import logging
import heapq
from similarity import DescSimilarity, TagSimilarity, TitleSimilarity, StarSimilarity
from structural_similarity import LinearStructuralSimilarity
from scipy.spatial.distance import cosine


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
			matrix_row = []
			heapq.heapify(matrix_row)
			s_matrix.append(matrix_row)

		for i, fir_video in enumerate(videos):
			print "compute similarity documents: " + str(i)
			idx_dict[i] = fir_video.vid
			#if fir_video.vid == '192':
			#	print 'name: '
			#	if fir_video.name is not None:
			#		ret = ''
			#		for item in fir_video.name:
			#			ret += item
			#		print ret.encode('utf-8', errors='ignore')
			#	else:
			#		print None
			#	print 'desc: '
			#	if fir_video.desc is not None:
			#		ret = ''
			#		for item in fir_video.desc:
			#			ret += item
			#		print ret.encode('utf-8', errors='ignore')
			#	else:
			#		print None
			#	print 'tag: '
			#	if fir_video.tags is not None:
			#		ret = ''
			#		for item in fir_video.tags:
			#			ret += item
			#		#print ret.encode('utf-8', errors='ignore')
			#		print ret
			#	else:
			#		print None
			#	print 'stars: '
			#	if fir_video.stars is not None:
			#		ret = ''
			#		for item in fir_video.stars:
			#			ret += item
			#		#print ret.encode('utf-8', errors='ignore')
			#		print ret
			#	else:
			#		print None

			for j, sec_video in enumerate(videos):
				if i < j:
					#desc_score = desc_similarity.compute(fir_video.desc, sec_video.desc, self.word2vec_model)
					#print doc_vector[i]
					#print doc_vector[j]
					# using scipy cosine distance, so compute similarity we have compute 1 - dt
					desc_score = 1 - cosine(doc_vector[i], doc_vector[j])
					#title_score = title_similarity.compute(fir_video.name, sec_video.name, self.word2vec_model)
					title_score = title_similarity.cosin_compute(fir_video.name, sec_video.name)
					tag_score = tag_similarity.compute(fir_video.tags, sec_video.tags)
					star_score = star_similarity.compute(fir_video.stars, sec_video.stars)
					#if fir_video.vid == '192':
					#	print sec_video.vid
					#	print 'name: '
					#	if sec_video.name is not None:
					#		ret = ''
					#		for item in sec_video.name:
					#			ret += item
					#		print ret.encode('utf-8', errors='ignore')
					#	else:
					#		print None
					#	print 'desc: '
					#	if sec_video.desc is not None:
					#		ret = ''
					#		for item in sec_video.desc:
					#			ret += item
					#		print ret.encode('utf-8', errors='ignore')
					#	else:
					#		print None
					#	print 'tag: '
					#	if sec_video.tags is not None:
					#		ret = ''
					#		for item in sec_video.tags:
					#			ret += item
					#		#print ret.encode('utf-8', errors='ignore')
					#		print ret
					#	else:
					#		print None
					#	print 'stars: '
					#	if sec_video.stars is not None:
					#		ret = ''
					#		for item in sec_video.stars:
					#			ret += item
					#		#print ret.encode('utf-8', errors='ignore')
					#		print ret

					#	print "desc: " + str(desc_score)
					#	print "title: " + str(title_score)
					#	print "tag: " + str(tag_score)
					#	print "star: " + str(star_score)
					linear_simialarity.set_similarity([desc_score, title_score, tag_score, star_score])
					s = linear_simialarity.compute()
					sp = ScorePair(j, s)
					if len(s_matrix[i]) < self.topN:
						s_matrix[i].append(sp)
					else:
						heapq.heappushpop(s_matrix[i], sp)
					if len(s_matrix[j]) < self.topN:
						s_matrix[j].append(sp)
					else:
						heapq.heappushpop(s_matrix[j], sp)



		for i, matrix_row in enumerate(s_matrix):
			s_matrix[i] = heapq.nlargest(self.topN, matrix_row)

		return (s_matrix, idx_dict)

	def save(self, s_matrix, idx_dict):
		with open(self.matrix_out, 'w') as m_o:
			for id, top_similarity in enumerate(s_matrix):
				if id in idx_dict:
					vid = idx_dict[id]
					ret_list = []
					for item in top_similarity:
						s_id = item.idx
						score = item.s
						if s_id in idx_dict:
							s_vid = idx_dict[s_id]
							ret_list.append(str(s_vid) + ':' + str(score))
					m_o.write(str(vid) + '\t' + ','.join(ret_list) + '\n')






