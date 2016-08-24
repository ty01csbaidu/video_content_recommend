#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: word2vec_learner.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/17 下午5:45

import logging
import sys
import os

from gensim.corpora import WikiCorpus
import jieba


class MySentences(object):

	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.strip(' ').split(' ')


class Learner(object):
	"""
	learning word2vec model from corpus
	"""
	def __init__(self, model_out, size, window, alpha, workers, sg, hs, negative, min_count):
		self.model_out = model_out
		self.size = size
		self.window = window
		self.alpha = alpha
		self.workers = workers
		self.sg = sg
		self.hs = hs
		self.negative = negative
		self.min_count = min_count

	def train(self, sentences):
		"""
		train model from sentences
		:return: model
		"""

logger = logging.getLogger('models.wikilearner')

class WikiLearner(Learner):

	def __init__(self, model_out, size, window, alpha, worker, sg, hs,
							 negative, min_count, zhwiki_out, simpleWiki_out, corpus_path):
		super(WikiLearner, self).__init__(model_out, size, window, alpha, worker, sg, hs,
																			negative, min_count)
		self.zhwiki_out = zhwiki_out
		self.simpleWiki_out = simpleWiki_out
		self.corpus_path = corpus_path

	def get_corpus(self):
		"""
		get chinese-wiki text
		translate to simple chinese
		:return:
		"""
		#os.system('wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2')
		wiki = WikiCorpus('zhwiki-latest-pages-articles.xml.bz2', lemmatize=False, dictionary={})
		for text in wiki.get_texts():
			self.zhwiki_out.write(" ".join(text) + "\n")
			i = i + 1
			if (i % 10000 == 0):
				logger.info("Saved " + str(i) + " articles")

		self.zhwiki_out.close()
		logger.info("Finished Saved " + str(i) + " articles")

		# translate to simple chinese
		os.system('opencc -i ' + self.zhwiki_out + ' -o ' + os.path.join(self.corpus_path, self.simpleWiki_out) + ' -c zht2zhs.ini')

		os.system('iconv -c -t utf-8 < ' + os.path.join(self.corpus_path, self.simpleWiki_out) + ' > ' + os.path.join(self.corpus_path, 'tmp'))
		#os.system('mv ' + os.path.join(self.corpus_path, 'tmp') + ' ' + os.path.join(self.corpus_path, self.simpleWiki_out))
		os.system('cat /dev/null > ' + os.path.join(self.corpus_path, self.simpleWiki_out))
		with open(os.path.join(self.corpus_path, self.simpleWiki_out), 'w') as wiki_token_out:
			with open(os.path.join(self.corpus_path, 'tmp'), 'r') as wiki_in:
				for line in wiki_in:
					seg_list = jieba.cut(line)
					wiki_token_out.write(' '.join(seg_list)+'\n')

	def token(self):
		"""
		iterable tokens
		:return:
		"""
		sentences = MySentences(self.corpus_path)
		return sentences

	def train(self, sentences):
		model = Word2vec(sentences, size = self.size, window = self.window, alpha = self.alpha,
										 workers = self.workers, sg = self.sg, hs = self.hs, negative = self.negative,
										 min_count = self.min_count)

		model.save_word2vec_format(self.model_out, binary=True)


if __name__ == '__main__':
	wiki_out = sys.argv[1]
	simpleWiki_out = sys.argv[2]
	corpus_path = sys.argv[3]
	model_out = sys.argv[4]
	size = 100
	window = 5
	alpha = 0.025
	workers = 4
	sg = 0
	hs = 1
	negative = 0
	min_count = 5

	wiki_learner = WikiLearner(model_out, size, window, alpha, workers, sg, hs, negative,
														 min_count, wiki_out, simpleWiki_out, corpus_path)
	wiki_learner.get_corpus()
	sentences = wiki_learner.token()
	wiki_learner.train(sentences)
