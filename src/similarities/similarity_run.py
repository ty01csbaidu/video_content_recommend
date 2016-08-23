#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: similarity_run.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 上午11:46


from similarity_matrix import SimilarityMatrix
from src.models.video import Video, VideoIterable

import sys
from gensim.models import Word2vec


if __name__ == '__main__':
	trained_model = sys.argv[1]
	vid_file = sys.argv[2]
	similarity_out = sys.argv[3]
	model = Word2vec.load_word2vec_format(trained_model, binary=True)
	videos = VideoIterable(vid_file)
	similarity_matrix = SimilarityMatrix(similarity_out, model, 20)
	s_matrix, idx_dict = similarity_matrix.compute(videos)
	similarity_matrix.save(s_matrix, idx_dict)

