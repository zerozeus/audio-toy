"""
There are 32 different Mobilenet models to choose from, with a variety of file
size and latency options. The first number can be '1.0', '0.75', '0.50', or
'0.25' to control the size, and the second controls the input image size, either
'224', '192', '160', or '128', with smaller sizes running faster. See
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
for more information on Mobilenet.

To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from bean.ml_config import MLConfig
from bean.audio_sample import AudioSample, parseLabelConfig
from util.parser import AudioParser

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2**27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):

	if not gfile.Exists(image_dir):
		tf.logging.error("Image directory '" + image_dir + "' not found.")
		return None
	result = collections.OrderedDict()
	sub_dirs = [
		os.path.join(image_dir, item)
		for item in gfile.ListDirectory(image_dir)
	]
	sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))
	for sub_dir in sub_dirs:
		extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		if dir_name == image_dir:
			continue
		tf.logging.info("Looking for images in '" + dir_name + "'")
		for extension in extensions:
			file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
			file_list.extend(gfile.Glob(file_glob))
		if not file_list:
			tf.logging.warning('No files found')
			continue
		if len(file_list) < 20:
			tf.logging.warning(
				'WARNING: Folder has less than 20 images, which may cause issues.'
			)
		elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
			tf.logging.warning(
				'WARNING: Folder {} has more than {} images. Some images will '
				'never be selected.'.format(dir_name,
											MAX_NUM_IMAGES_PER_CLASS))
		label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
		training_images = []
		testing_images = []
		validation_images = []
		for file_name in file_list:
			base_name = os.path.basename(file_name)
			# We want to ignore anything after '_nohash_' in the file name when
			# deciding which set to put an image in, the data set creator has a way of
			# grouping photos that are close variations of each other. For example
			# this is used in the plant disease data set to group multiple pictures of
			# the same leaf.
			hash_name = re.sub(r'_nohash_.*$', '', file_name)
			# This looks a bit magical, but we need to decide whether this file should
			# go into the training, testing, or validation sets, and we want to keep
			# existing files in the same set even if more files are subsequently
			# added.
			# To do that, we need a stable way of deciding based on just the file name
			# itself, so we do a hash of that and then use that to generate a
			# probability value that we use to assign it.
			hash_name_hashed = hashlib.sha1(
				compat.as_bytes(hash_name)).hexdigest()
			percentage_hash = (
				(int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
				(100.0 / MAX_NUM_IMAGES_PER_CLASS))
			if percentage_hash < validation_percentage:
				validation_images.append(base_name)
			elif percentage_hash < (
					testing_percentage + validation_percentage):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)
		result[label_name] = {
			'dir': dir_name,
			'training': training_images,
			'testing': testing_images,
			'validation': validation_images,
		}
	return result


def get_image_path(image_lists, label_name, index, image_dir, category):

	if label_name not in image_lists:
		tf.logging.fatal('Label does not exist %s.', label_name)
	label_lists = image_lists[label_name]
	if category not in label_lists:
		tf.logging.fatal('Category does not exist %s.', category)
	category_list = label_lists[category]
	if not category_list:
		tf.logging.fatal('Label %s has no images in the category %s.',
						 label_name, category)
	mod_index = index % len(category_list)
	base_name = category_list[mod_index]
	sub_dir = label_lists['dir']
	full_path = os.path.join(image_dir, sub_dir, base_name)
	return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
						category, architecture):
	return get_image_path(image_lists, label_name, index, bottleneck_dir,
						  category) + '_' + architecture + '.txt'


def create_model_graph(model_info, config):

	with tf.Graph().as_default() as graph:
		model_path = os.path.join(config['model_dir'],
								  model_info['model_file_name'])
		with gfile.FastGFile(model_path, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
				graph_def,
				name='',
				return_elements=[
					model_info['bottleneck_tensor_name'],
					model_info['resized_input_tensor_name'],
				]))
	return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
							decoded_image_tensor, resized_input_tensor,
							bottleneck_tensor):

	# First decode the JPEG image, resize it, and rescale the pixel values.
	resized_input_values = sess.run(decoded_image_tensor,
									{image_data_tensor: image_data})
	# Then run it through the recognition network.
	bottleneck_values = sess.run(bottleneck_tensor,
								 {resized_input_tensor: resized_input_values})
	bottleneck_values = np.squeeze(bottleneck_values)
	return bottleneck_values


def maybe_download_and_extract(config, data_url):
	
	dest_directory = config['model_dir']
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = data_url.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):

		def _progress(count, block_size, total_size):
			sys.stdout.write(
				'\r>> Downloading %s %.1f%%' %
				(filename,
				 float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
		statinfo = os.stat(filepath)
		tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
						'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
	"""Makes sure the folder exists on disk.

	Args:
		dir_name: Path string to the folder we want to create.
	"""
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
						   image_dir, category, sess, jpeg_data_tensor,
						   decoded_image_tensor, resized_input_tensor,
						   bottleneck_tensor):
	"""Create a single bottleneck file."""
	tf.logging.info('Creating bottleneck at ' + bottleneck_path)
	image_path = get_image_path(image_lists, label_name, index, image_dir,
								category)
	if not gfile.Exists(image_path):
		tf.logging.fatal('File does not exist %s', image_path)
	image_data = gfile.FastGFile(image_path, 'rb').read()
	try:
		bottleneck_values = run_bottleneck_on_image(
			sess, image_data, jpeg_data_tensor, decoded_image_tensor,
			resized_input_tensor, bottleneck_tensor)
	except Exception as e:
		raise RuntimeError(
			'Error during processing file %s (%s)' % (image_path, str(e)))
	bottleneck_string = ','.join(str(x) for x in bottleneck_values)
	with open(bottleneck_path, 'w') as bottleneck_file:
		bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
							 category, bottleneck_dir, jpeg_data_tensor,
							 decoded_image_tensor, resized_input_tensor,
							 bottleneck_tensor, architecture):
   
	label_lists = image_lists[label_name]
	sub_dir = label_lists['dir']
	sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
	ensure_dir_exists(sub_dir_path)
	bottleneck_path = get_bottleneck_path(
		image_lists, label_name, index, bottleneck_dir, category, architecture)
	if not os.path.exists(bottleneck_path):
		create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
							   image_dir, category, sess, jpeg_data_tensor,
							   decoded_image_tensor, resized_input_tensor,
							   bottleneck_tensor)
	with open(bottleneck_path, 'r') as bottleneck_file:
		bottleneck_string = bottleneck_file.read()
	did_hit_error = False
	try:
		bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
	except ValueError:
		tf.logging.warning('Invalid float found, recreating bottleneck')
		did_hit_error = True
	if did_hit_error:
		create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
							   image_dir, category, sess, jpeg_data_tensor,
							   decoded_image_tensor, resized_input_tensor,
							   bottleneck_tensor)
		with open(bottleneck_path, 'r') as bottleneck_file:
			bottleneck_string = bottleneck_file.read()
		# Allow exceptions to propagate here, since they shouldn't happen after a
		# fresh creation
		bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
	return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
					  jpeg_data_tensor, decoded_image_tensor,
					  resized_input_tensor, bottleneck_tensor, architecture):
   
	how_many_bottlenecks = 0
	ensure_dir_exists(bottleneck_dir)
	for label_name, label_lists in image_lists.items():
		for category in ['training', 'testing', 'validation']:
			category_list = label_lists[category]
			for index, unused_base_name in enumerate(category_list):
				get_or_create_bottleneck(
					sess, image_lists, label_name, index, image_dir, category,
					bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
					resized_input_tensor, bottleneck_tensor, architecture)

				how_many_bottlenecks += 1
				if how_many_bottlenecks % 100 == 0:
					tf.logging.info(
						str(how_many_bottlenecks) +
						' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
								  bottleneck_dir, image_dir, jpeg_data_tensor,
								  decoded_image_tensor, resized_input_tensor,
								  bottleneck_tensor, architecture):
 
	class_count = len(image_lists.keys())
	bottlenecks = []
	ground_truths = []
	filenames = []
	if how_many >= 0:
		# Retrieve a random sample of bottlenecks.
		for unused_i in range(how_many):
			label_index = random.randrange(class_count)
			label_name = list(image_lists.keys())[label_index]
			image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
			image_name = get_image_path(image_lists, label_name, image_index,
										image_dir, category)
			bottleneck = get_or_create_bottleneck(
				sess, image_lists, label_name, image_index, image_dir,
				category, bottleneck_dir, jpeg_data_tensor,
				decoded_image_tensor, resized_input_tensor, bottleneck_tensor,
				architecture)
			ground_truth = np.zeros(class_count, dtype=np.float32)
			ground_truth[label_index] = 1.0
			bottlenecks.append(bottleneck)
			ground_truths.append(ground_truth)
			filenames.append(image_name)
	else:
		# Retrieve all bottlenecks.
		for label_index, label_name in enumerate(image_lists.keys()):
			for image_index, image_name in enumerate(
					image_lists[label_name][category]):
				image_name = get_image_path(image_lists, label_name,
											image_index, image_dir, category)
				bottleneck = get_or_create_bottleneck(
					sess, image_lists, label_name, image_index, image_dir,
					category, bottleneck_dir, jpeg_data_tensor,
					decoded_image_tensor, resized_input_tensor,
					bottleneck_tensor, architecture)
				ground_truth = np.zeros(class_count, dtype=np.float32)
				ground_truth[label_index] = 1.0
				bottlenecks.append(bottleneck)
				ground_truths.append(ground_truth)
				filenames.append(image_name)
	return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
		sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
		distorted_image, resized_input_tensor, bottleneck_tensor):

	class_count = len(image_lists.keys())
	bottlenecks = []
	ground_truths = []
	for unused_i in range(how_many):
		label_index = random.randrange(class_count)
		label_name = list(image_lists.keys())[label_index]
		image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
		image_path = get_image_path(image_lists, label_name, image_index,
									image_dir, category)
		if not gfile.Exists(image_path):
			tf.logging.fatal('File does not exist %s', image_path)
		jpeg_data = gfile.FastGFile(image_path, 'rb').read()
		# Note that we materialize the distorted_image_data as a numpy array before
		# sending running inference on the image. This involves 2 memory copies and
		# might be optimized in other implementations.
		distorted_image_data = sess.run(distorted_image,
										{input_jpeg_tensor: jpeg_data})
		bottleneck_values = sess.run(
			bottleneck_tensor, {resized_input_tensor: distorted_image_data})
		bottleneck_values = np.squeeze(bottleneck_values)
		ground_truth = np.zeros(class_count, dtype=np.float32)
		ground_truth[label_index] = 1.0
		bottlenecks.append(bottleneck_values)
		ground_truths.append(ground_truth)
	return bottlenecks, ground_truths


def add_input_distortions(random_crop, random_scale, input_width, input_height,
						  input_depth, input_mean, input_std):
	jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
	decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	margin_scale = 1.0 + (random_crop / 100.0)
	resize_scale = 1.0 + (random_scale / 100.0)
	margin_scale_value = tf.constant(margin_scale)
	resize_scale_value = tf.random_uniform(
		tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
	scale_value = tf.multiply(margin_scale_value, resize_scale_value)
	precrop_width = tf.multiply(scale_value, input_width)
	precrop_height = tf.multiply(scale_value, input_height)
	precrop_shape = tf.stack([precrop_height, precrop_width])
	precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
	precropped_image = tf.image.resize_bilinear(decoded_image_4d,
												precrop_shape_as_int)
	precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
	cropped_image = tf.random_crop(precropped_image_3d,
								   [input_height, input_width, input_depth])

	offset_image = tf.subtract(cropped_image, input_mean)
	mul_image = tf.multiply(offset_image, 1.0 / input_std)
	distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
	return jpeg_data, distort_result


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
						   bottleneck_tensor_size, learning_rate):
	with tf.name_scope('input'):
		bottleneck_input = tf.placeholder_with_default(
			bottleneck_tensor,
			shape=[None, bottleneck_tensor_size],
			name='BottleneckInputPlaceholder')

		ground_truth_input = tf.placeholder(
			tf.float32, [None, class_count], name='GroundTruthInput')

	# Organizing the following ops as `final_training_ops` so they're easier
	# to see in TensorBoard
	layer_name = 'final_training_ops'
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			initial_value = tf.truncated_normal(
				[bottleneck_tensor_size, class_count], stddev=0.001)

			layer_weights = tf.Variable(initial_value, name='final_weights')

			variable_summaries(layer_weights)
		with tf.name_scope('biases'):
			layer_biases = tf.Variable(
				tf.zeros([class_count]), name='final_biases')
			variable_summaries(layer_biases)
		with tf.name_scope('Wx_plus_b'):
			logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
			tf.summary.histogram('pre_activations', logits)

	final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
	tf.summary.histogram('activations', final_tensor)

	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
			labels=ground_truth_input, logits=logits)
		with tf.name_scope('total'):
			cross_entropy_mean = tf.reduce_mean(cross_entropy)
	tf.summary.scalar('cross_entropy', cross_entropy_mean)

	with tf.name_scope('train'):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		train_step = optimizer.minimize(cross_entropy_mean)

	return (train_step, cross_entropy_mean, bottleneck_input,
			ground_truth_input, final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			prediction = tf.argmax(result_tensor, 1)
			correct_prediction = tf.equal(prediction,
										  tf.argmax(ground_truth_tensor, 1))
		with tf.name_scope('accuracy'):
			evaluation_step = tf.reduce_mean(
				tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', evaluation_step)
	return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name, final_tensor_name):
	output_graph_def = graph_util.convert_variables_to_constants(
		sess, graph.as_graph_def(), [final_tensor_name])
	with gfile.FastGFile(graph_file_name, 'wb') as f:
		f.write(output_graph_def.SerializeToString())
	return


def prepare_file_system(config):
	# Setup the directory we'll write summaries to for TensorBoard
	if tf.gfile.Exists(config['summaries_dir']):
		tf.gfile.DeleteRecursively(config['summaries_dir'])
	tf.gfile.MakeDirs(config['summaries_dir'])
	if int(config['intermediate_store_frequency']) > 0:
		ensure_dir_exists(config['intermediate_output_graphs_dir'])
	return

def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):

	jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
	decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	resize_shape = tf.stack([input_height, input_width])
	resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
	resized_image = tf.image.resize_bilinear(decoded_image_4d,
											 resize_shape_as_int)
	offset_image = tf.subtract(resized_image, input_mean)
	mul_image = tf.multiply(offset_image, 1.0 / input_std)

	return jpeg_data, mul_image

def train(args):
	if not args.image_dir or not args.output_graph or not args.output_labels or not args.config:
		tf.logging.error('Wrong necessary parameters')
		return -1
	parent = os.path.join(os.getcwd(), 'src/ml')
	config = MLConfig(parent, os.path.join(parent, args.config))
	config.add_params('image_dir', args.image_dir, param_type="str")
	config.add_params('output_graph', args.output_graph)
	config.add_params('output_labels', args.output_labels)

	output_dir = os.path.dirname(config['output_graph'])
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	tf.logging.set_verbosity(tf.logging.INFO)
	prepare_file_system(config)
	model_info = MLConfig.create_model_info(config['architecture'])

	if not model_info:
		tf.logging.error('Did not recognize architecture flag')
		return -1

	maybe_download_and_extract(config, model_info['data_url'])
	graph, bottleneck_tensor, resized_image_tensor = create_model_graph(model_info, config)
	image_lists = create_image_lists(config['image_dir'],
									 config['testing_percentage'],
									 config['validation_percentage'])
	class_count = len(image_lists.keys())
	if class_count == 0:
		tf.logging.error('No valid folders of images found at ' +
						 config['image_dir'])
		return -1
	if class_count == 1:
		tf.logging.error('Only one valid folder of images found at ' +
						 config['image_dir'] +
						 ' - multiple classes are needed for classification.')
		return -1

	do_distort_images = (config['random_crop'] != 0) or (config['random_scale'] != 0)
	
	with tf.Session(graph=graph) as sess:

		jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
			model_info['input_width'], model_info['input_height'],
			model_info['input_depth'], model_info['input_mean'],
			model_info['input_std'])
		if do_distort_images:
			(distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(
					config['random_crop'], config['random_scale'], 
					model_info['input_width'], model_info['input_height'], model_info['input_depth'],
					model_info['input_mean'], model_info['input_std'])
		else:
			# We'll make sure we've calculated the 'bottleneck' image summaries and
			# cached them on disk.
			cache_bottlenecks(sess, image_lists, config['image_dir'],
							  config['bottleneck_dir'], jpeg_data_tensor,
							  decoded_image_tensor, resized_image_tensor,
							  bottleneck_tensor, config['architecture'])

		# Add the new layer that we'll be training.
		(train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(
			 len(image_lists.keys()), config['final_tensor_name'],
			 bottleneck_tensor, model_info['bottleneck_tensor_size'], config['learning_rate'])

		# Create the operations we need to evaluate the accuracy of our new layer.
		evaluation_step, prediction = add_evaluation_step(
			final_tensor, ground_truth_input)

		# Merge all the summaries and write them out to the summaries_dir
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(config['summaries_dir'] + '/train',
											 sess.graph)

		validation_writer = tf.summary.FileWriter(config['summaries_dir'] +
												  '/validation')

		# Set up all our weights to their initial default values.
		init = tf.global_variables_initializer()
		sess.run(init)

		# Run the training for as many cycles as requested on the command line.
		for i in range(config['how_many_training_steps']):
			# Get a batch of input bottleneck values, either calculated fresh every
			# time with distortions applied, or from the cache stored on disk.
			if do_distort_images:
				(train_bottlenecks,
				 train_ground_truth) = get_random_distorted_bottlenecks(
					 sess, image_lists, config['train_batch_size'], 'training',
					 config['image_dir'], distorted_jpeg_data_tensor,
					 distorted_image_tensor, resized_image_tensor,
					 bottleneck_tensor)
			else:
				(train_bottlenecks,
				 train_ground_truth, _) = get_random_cached_bottlenecks(
					 sess, image_lists, config['train_batch_size'], 'training',
					 config['bottleneck_dir'], config['image_dir'], jpeg_data_tensor,
					 decoded_image_tensor, resized_image_tensor,
					 bottleneck_tensor, config['architecture'])
			# Feed the bottlenecks and ground truth into the graph, and run a training
			# step. Capture training summaries for TensorBoard with the `merged` op.
			train_summary, _ = sess.run(
				[merged, train_step],
				feed_dict={
					bottleneck_input: train_bottlenecks,
					ground_truth_input: train_ground_truth
				})
			train_writer.add_summary(train_summary, i)

			# Every so often, print out how well the graph is training.
			is_last_step = (i + 1 == config['how_many_training_steps'])
			if (i % config['eval_step_interval']) == 0 or is_last_step:
				train_accuracy, cross_entropy_value = sess.run(
					[evaluation_step, cross_entropy],
					feed_dict={
						bottleneck_input: train_bottlenecks,
						ground_truth_input: train_ground_truth
					})
				tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
								(datetime.now(), i, train_accuracy * 100))
				tf.logging.info('%s: Step %d: Cross entropy = %f' %
								(datetime.now(), i, cross_entropy_value))
				validation_bottlenecks, validation_ground_truth, _ = (
					get_random_cached_bottlenecks(
						sess, image_lists, config['validation_batch_size'],
						'validation', config['bottleneck_dir'], config['image_dir'],
						jpeg_data_tensor, decoded_image_tensor,
						resized_image_tensor, bottleneck_tensor,
						config['architecture']))
				# Run a validation step and capture training summaries for TensorBoard
				# with the `merged` op.
				validation_summary, validation_accuracy = sess.run(
					[merged, evaluation_step],
					feed_dict={
						bottleneck_input: validation_bottlenecks,
						ground_truth_input: validation_ground_truth
					})
				validation_writer.add_summary(validation_summary, i)
				tf.logging.info(
					'%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
					(datetime.now(), i, validation_accuracy * 100,
					 len(validation_bottlenecks)))

			# Store intermediate results
			intermediate_frequency = config['intermediate_store_frequency']

			if (intermediate_frequency > 0
					and (i % intermediate_frequency == 0) and i > 0):
				intermediate_file_name = (config['intermediate_output_graphs_dir']
										  + 'intermediate_' + str(i) + '.pb')
				tf.logging.info('Save intermediate result to : ' +
								intermediate_file_name)
				save_graph_to_file(sess, graph, intermediate_file_name, config['final_tensor_name'])

		# We've completed all our training, so run a final test evaluation on
		# some new images we haven't used before.
		test_bottlenecks, test_ground_truth, test_filenames = (
			get_random_cached_bottlenecks(
				sess, image_lists, config['test_batch_size'], 'testing',
				config['bottleneck_dir'], config['image_dir'], jpeg_data_tensor,
				decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
				config['architecture']))
		test_accuracy, predictions = sess.run(
			[evaluation_step, prediction],
			feed_dict={
				bottleneck_input: test_bottlenecks,
				ground_truth_input: test_ground_truth
			})
		tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))
		tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
		for i, test_filename in enumerate(test_filenames):
			if predictions[i] != test_ground_truth[i].argmax():
				tf.logging.info( '%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))
		
		save_graph_to_file(sess, graph, config['output_graph'], config['final_tensor_name'])
		with gfile.FastGFile(config['output_labels'], 'w') as f:
			f.write('\n'.join(image_lists.keys()) + '\n')

def prepare_input():
	parent = os.path.abspath(os.getcwd())
	audio_enum = parseLabelConfig(os.path.join(parent, "data/config.xml"))
	# parser.to_image(AudioSample(os.path.join(parent, "data/test.xml"), audio_enum), os.path.join(parent, "data/image/test"), trim = False)
	return AudioParser().to_image(AudioSample(os.path.join(parent, "data/train.xml"), audio_enum), os.path.join(parent, "data/image/train"))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--image_dir',
		type=str,
		default='',
		help='Path to folders of labeled images.')
	parser.add_argument(
		'--output_graph',
		type=str,
		default='model/output_graph.pb',
		help='Where to save the trained graph.')

	parser.add_argument(
		'--output_labels',
		type=str,
		default='model/output_labels.txt',
		help='Where to save the trained graph\'s labels.')

	parser.add_argument(
		'--config',
		type=str,
		default='train_config.xml',
		help='Store all necessary training params.')

	args = parser.parse_args()

	if not args.image_dir:
		args.image_dir = prepare_input()

	train(args)
