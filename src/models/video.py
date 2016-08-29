#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @file: video.py
# @author: tangye <tangye@mgtv.com|tangyel@126.com>
# @time: 16/8/22 上午11:59


import logging
import jieba
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
		fields = self.line.strip('\t').strip('\n').split('\t')
		#print self.line
		if len(fields) >= 3:
			self.vid = fields[0]
			name = fields[1].strip(' ').split(' ')
			if len(name) > 0:
				self.name = [item.decode('utf-8') for item in name]
				#self.name = name
			self.cid = fields[2]

		if len(fields) >= 5:
			desc = fields[4].strip(' ').split(' ')
			desc = [item.decode('utf-8') for item in desc if item not in filtered_set]
			#desc = [item for item in desc if item not in filtered_set]
			if len(desc) > 0:
				self.desc = desc

		if len(fields) >= 6:
			if len(fields[5]) > 0:
				tags = fields[5].strip(' ').split(',')
				#print tags
				if len(tags) > 0 and not tags == ['']:
					self.tags = set(tags)

		if len(fields) >= 8:
			if len(fields[7]) > 0:
				stars = fields[7].strip(' ').split('|')
				#print stars
				if len(stars) > 0 and not stars == ['']:
					self.stars = set(stars)

		return self


class VideoIterable(object):

	def __init__(self, input_file):
		self.input_file = input_file

	def __iter__(self):
		with open(self.input_file) as i_f:
			for line in i_f:
				video = Video(line)
				yield video.read()


def video_token(vid_file, vid_out, corpus_out):
	with open(corpus_out, 'w') as c_o:
		with open(vid_out, 'w') as v_o:
			with open(vid_file, 'r') as v_f:
				for line in v_f:
					fields = line.strip('\t').strip('\n').split('\t')
					if len(fields) >= 3:
						name = fields[1].strip(' ')
						name_seg_list = jieba.cut(name)
						tokenized_name = ' '.join(name_seg_list)
						fields[1] = tokenized_name.encode('utf-8')
						#c_o.write((tokenized_name+'\n').encode('utf-8'))

					if len(fields) >= 5:
						desc = fields[4].strip(' ')
						desc_seg_list = jieba.cut(desc)
						tokenized_desc = ' '.join(desc_seg_list)
						c_o.write((tokenized_desc+'\n').encode('utf-8'))
						fields[4] = tokenized_desc.encode('utf-8')

					v_o.write('\t'.join(fields)+'\n')


if __name__ == '__main__':
	vid_file = sys.argv[1]
	vid_out = sys.argv[2]
	corpus_out = sys.argv[3]
	video_token(vid_file, vid_out, corpus_out)









