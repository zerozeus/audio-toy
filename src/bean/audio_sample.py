#!/usr/bin/python
# -*- coding: UTF-8 -*-

import xml.etree.ElementTree as ET
from enum import Enum
import os

class AudioSample(object):
	
	def __init__(self, path, audio_type):
		self.audio_enum = audio_type
		self.data_dir, self.data_name = os.path.split(path)
		self.audio_dir, self.content = self.__parseFromFile__(path, audio_type)

	def __parseFromFile__(self, path, audio_type):
		tree = ET.ElementTree(file=path)
		root = tree.getroot()
		audio_dir = root.attrib["audio_dir"]
		audio_files = {}
		for child in root:
			audio_file_attribute = AudioSampleAttribute(child.attrib["audio_file"])
			audio_file_content = []
			for content in child:
				audio_file_content.append(AudioSampleContent(audio_type(int(content.attrib["type"])), content.attrib["start"], content.attrib["end"]))
			audio_files[audio_file_attribute] = audio_file_content

		return os.path.join(self.data_dir, audio_dir), audio_files

class AudioSampleAttribute(object):
	def __init__(self, file_name):
		self.sub_dir, self.file_name = os.path.split(file_name)
	def __str__(self):
		return "[file_name: {}]".format(self.file_name)


class AudioSampleContent(object):
	def __init__(self, audio_type, start, end):
		self.type = audio_type
		self.start = int(start)
		self.end = int(end)
	def __str__(self):
		return "[type: {}], [start: {}], [end: {}]".format(self.type, self.start, self.end)
	
	def duration(self):
		return self.end - self.start

def parseLabelConfig(path):
	tree = ET.ElementTree(file=path)
	config = {}
	for child in tree.getroot():
		config[child.attrib['name']] = int(child.text)
	return Enum("AudioType", config)















