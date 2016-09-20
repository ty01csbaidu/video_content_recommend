#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_run.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 上午11:46


from similarity_matrix import SimilarityMatrix

import sys
import os
import time
from gensim.models import Word2Vec
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from models.video import Video, VideoIterable


if __name__ == '__main__':
	trained_model = sys.argv[1]
	old_vid_file = sys.argv[2]
	old_similarity_file = sys.argv[3]
	old_doc_vector_file = sys.argv[4]
	vid_file = sys.argv[5]
	matrix_out = sys.argv[6]
	doc_vector_out = sys.argv[7]

	start_time = time.time()
	model = Word2Vec.load_word2vec_format(trained_model, binary=True)
	print("word2vec model load: %s" % (time.time() - start_time))
	old_videos = VideoIterable(old_vid_file)
	videos = VideoIterable(vid_file)
	similarity_matrix = SimilarityMatrix(model, 20)
	similarity_matrix.run(old_similarity_file, old_doc_vector_file, old_videos, matrix_out, doc_vector_out, videos)

