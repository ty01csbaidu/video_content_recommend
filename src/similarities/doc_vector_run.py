#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: doc_vector_run.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/9/20 下午3:40


from similarity_matrix import SimilarityMatrix

import sys
import os
import time
from gensim.models import Word2Vec
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from models.video import Video, VideoIterable


if __name__ == '__main__':
	trained_model = sys.argv[1]
	vid_file = sys.argv[2]
	doc_vector_out = sys.argv[3]

	start_time = time.time()
	model = Word2Vec.load_word2vec_format(trained_model, binary=True)
	print("word2vec model load: %s" % (time.time() - start_time))
	videos = VideoIterable(vid_file)
	similarity_matrix = SimilarityMatrix(model, 20)
	similarity_matrix.compute_doc_vector(videos)
	similarity_matrix.save_doc_vector(doc_vector_out)
