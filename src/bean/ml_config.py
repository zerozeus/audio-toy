import xml.etree.ElementTree as ET
from enum import Enum
import os
import tensorflow as tf

class MLConfig(object):
	
	def __init__(self, parent_path, config_path):
		self.config_path = config_path
		self.parent_path = parent_path
		self.content = self.__parseFromFile__(config_path)

	def __parseFromFile__(self, path):
		tree = ET.ElementTree(file=path)
		root = tree.getroot()
		params = {}
		for param in root.findall('param'):
			param_type = param.attrib['type']
			value = param.attrib['value']
			key = param.attrib['name']
			params[key] = self.__parseValue__(value, param_type)
		return params

	def __getitem__(self, key):
		return self.content[key]

	def __parseValue__(self, value, param_type):
		if param_type == 'int':
			value = int(value)
		elif param_type == 'float':
			value = float(value)
		elif param_type == 'long':
			value = long(value)
		elif param_type == 'path':
			value = os.path.join(self.parent_path, value)
		return value

	def add_params(self, key, value, param_type='path'):
		if not value or not key:
			return
		self.content[key] = self.__parseValue__(value, param_type)

	@staticmethod
	def create_model_info(architecture):
		architecture = architecture.lower()
		if architecture == 'inception_v3':
			# pylint: disable=line-too-long
			data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
			# pylint: enable=line-too-long
			bottleneck_tensor_name = 'pool_3/_reshape:0'
			bottleneck_tensor_size = 2048
			input_width = 299
			input_height = 299
			input_depth = 3
			resized_input_tensor_name = 'Mul:0'
			model_file_name = 'classify_image_graph_def.pb'
			input_mean = 128
			input_std = 128
		elif architecture.startswith('mobilenet_'):
			parts = architecture.split('_')
			if len(parts) != 3 and len(parts) != 4:
				tf.logging.error("Couldn't understand architecture name '%s'",
								architecture)
				return None
			version_string = parts[1]
			if (version_string != '1.0' and version_string != '0.75'
					and version_string != '0.50' and version_string != '0.25'):
				tf.logging.error(
					""""The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
					but found '%s' for architecture '%s'""", version_string, architecture)
				return None
			size_string = parts[2]
			if (size_string != '224' and size_string != '192'
					and size_string != '160' and size_string != '128'):
				tf.logging.error(
					"""The Mobilenet input size should be '224', '192', '160', or '128',
					but found '%s' for architecture '%s'""", size_string, architecture)
				return None
			if len(parts) == 3:
				is_quantized = False
			else:
				if parts[3] != 'quantized':
					tf.logging.error(
						"Couldn't understand architecture suffix '%s' for '%s'",
						parts[3], architecture)
					return None
				is_quantized = True
			data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
			data_url += version_string + '_' + size_string + '_frozen.tgz'
			bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
			bottleneck_tensor_size = 1001
			input_width = int(size_string)
			input_height = int(size_string)
			input_depth = 3
			resized_input_tensor_name = 'input:0'
			if is_quantized:
				model_base_name = 'quantized_graph.pb'
			else:
				model_base_name = 'frozen_graph.pb'
			model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
			model_file_name = os.path.join(model_dir_name, model_base_name)
			input_mean = 127.5
			input_std = 127.5
		else:
			tf.logging.error("Couldn't understand architecture name '%s'",
							architecture)
			raise ValueError('Unknown architecture', architecture)

		return {
			'data_url': data_url,
			'bottleneck_tensor_name': bottleneck_tensor_name,
			'bottleneck_tensor_size': bottleneck_tensor_size,
			'input_width': input_width,
			'input_height': input_height,
			'input_depth': input_depth,
			'resized_input_tensor_name': resized_input_tensor_name,
			'model_file_name': model_file_name,
			'input_mean': input_mean,
			'input_std': input_std,
		}
		








