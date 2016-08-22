#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: video.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 上午11:59


import logging
import sys

logger = logging.getLogger('models.video')

filtered_set = {
	'无', '暂无'
}


class Video(object):
	"""
	read tokenized text line
		id \t 2016 超级女声 \t 83 \t 2016 蒙牛 酸酸乳 超级女声 ... \t 选秀,内地,音乐 \t 王心凌|古巨基
	"""
	def	__init__(self, line):
		self.line = line
		self.vid = None
		self.name = None
		self.cid = None
		self.desc = None
		self.stars = None
		self.tag = None

	def read(self):
		fields = self.line.strip('\t').split('\t')
		if len(fields) >= 3:
			self.vid = fields[0]
			name = fields[1].strip(' ').split(' ')
			if len(name) > 0:
				self.name = name
			self.cid = fields[2]

		if len(fields) >= 5:
			desc = fields[5].strip(' ').split(' ')
			if len(desc) > 0 and not desc in filtered_set:
				self.desc = desc

		if len(fields) >= 6:
			tags = fields[6].strip(' ').split(',')
			if len(tags) > 0:
				self.tags = tags

		if len(fields) >= 8:
			stars = fields[8].strip(' ').split('|')
			if len(stars) > 0:
				self.stars = stars

		return self


class VideoIterable(object):

	def __init__(self, input_file):
		self.input_file = input_file

	def __iter__(self):
		with open(self.input_file) as i_f:
			for line in i_f:
				video = Video(line)
				yield video.read()









