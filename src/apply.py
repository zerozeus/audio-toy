import glob, os
import argparse
import tensorflow as tf
import numpy as np

from bean.ml_config import MLConfig
from bean.audio_sample import AudioSample, parseLabelConfig
from util.parser import AudioParser

def load_graph(model_file):
	graph = tf.Graph()
	graph_def = tf.GraphDef()

	with open(model_file, "rb") as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)

	return graph

def load_labels(label_file):
	label = []
	proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
	for l in proto_as_ascii_lines:
		label.append(l.rstrip())
	return label

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255, input_channels=3):
	input_name = "file_reader"
	output_name = "normalized"
	file_reader = tf.read_file(file_name, input_name)
	image_reader = tf.image.decode_jpeg(file_reader, channels = input_channels,
											name='jpeg_reader')
	float_caster = tf.cast(image_reader, tf.float32)
	dims_expander = tf.expand_dims(float_caster, 0)
	resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
	normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
	sess = tf.Session()
	result = sess.run(normalized)

	return result


def apply(config):
	model_info = MLConfig.create_model_info(config['architecture'])
	input_name = model_info['resized_input_tensor_name'].split(':')[0]
	output_name = config['final_tensor_name']

	graph = load_graph(config['graph'])
	labels = load_labels(config['labels'])
	input_name = "import/" + input_name
	output_name = "import/" + output_name
	input_operation = graph.get_operation_by_name(input_name)
	output_operation = graph.get_operation_by_name(output_name)

	tf.logging.info("start parsing audio to image, audio path: {}, start: {}, end: {}".format(config['audio'], config['start'], config['end']))
	out_dir = AudioParser().audio_to_image(config['audio'], config['start'], config['end'], config['temp_image_location'])
	images = sorted(glob.glob(os.path.join(out_dir, "*.jpg")), key=os.path.getmtime)

	for index, image in enumerate(images) :
		t = read_tensor_from_image_file(image,
										input_height=model_info['input_height'],
										input_width=model_info['input_width'],
										input_mean=model_info['input_mean'],
										input_std=model_info['input_std'])

		with tf.Session(graph=graph) as sess:
			results = sess.run(output_operation.outputs[0],
							{input_operation.outputs[0]: t})
		results = np.squeeze(results)
		output_format = "completed: {}, image: {} (predictions={})"
		tf.logging.info(output_format.format(index*100.0/len(images), image, dict(zip(labels, results))))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--audio", 
		type=str,
		default='',
		help="audio to detect")
	parser.add_argument(
		"--start", 
		type=int,
		default=100,
		help="human voice detection start time(ms)")

	parser.add_argument(
		"--end", 
		type=int,
		default=5000,
		help="human voice detection end time(ms)")

	parser.add_argument(
		"--temp_image_location",
		type=str,
		default="apply/temp_image/"
	)

	parser.add_argument(
		"--graph", 
		type=str,
		default="model/output_graph.pb",
		help="graph/model to be executed")

	parser.add_argument(
		"--labels", 
		type=str,
		default="model/output_labels.txt",
		help="name of file containing labels")

	parser.add_argument(
		'--config',
		type=str,
		default='train_config.xml',
		help='Store all necessary training params.')
		
	tf.logging.set_verbosity(tf.logging.INFO)

	parent = os.path.join(os.getcwd(), 'src/ml')
	args = parser.parse_args()
	config = MLConfig(parent, os.path.join(parent, args.config))
	if not args.audio:
		tf.logging.error("empty audio path")
	else:
		config.add_params("graph", args.graph)
		config.add_params("labels", args.labels)
		config.add_params("temp_image_location", args.temp_image_location)
		config.add_params("audio", args.audio, param_type="str")
		config.add_params("start", args.start, param_type="int")
		config.add_params("end", args.end, param_type="int")
		apply(config)
