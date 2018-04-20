#!/usr/bin/python

from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf
import chess

FEATURE_LAYERS = (
	6 + # Friendly: pawns, knights, bishops, rooks, queens, king.
	6 + # Opponent: pawns, knights, bishops, rooks, queens, king.
	1)  # All ones for seeing the edge of the board in convolutions.

PIECE_NAMES = ["pawns", "knights", "bishops", "rooks", "queens", "kings"]

def extract_features(board):
	features = np.zeros((8, 8, FEATURE_LAYERS), dtype=np.int8)
	white_to_move = board.turn

	# Iterate over piece kinds, writing each kind into the feature map.
	for piece_index, name in enumerate(PIECE_NAMES):
		occupancy = getattr(board, name)
		# Iterate over possible board locations at which this piece could exist.
		for square_index in range(64):
			square_mask = 1 << square_index
			if occupancy & square_mask:
				# If a piece of kind `piece_index` does indeed exist at `square_index`, figure out it color and insert it.
				piece_is_white = bool(board.occupied_co[chess.WHITE] & square_mask)
				if not piece_is_white:
					assert board.occupied_co[chess.BLACK] & square_mask
				assert white_to_move in (False, True) and piece_is_white in (False, True)
				piece_is_friendly = white_to_move == piece_is_white
				features[
					square_index // 8,
					square_index % 8,
					piece_index + (1 - piece_is_friendly) * 6,
				] = 1

	# If we're encoding a move for black then we flip the board vertically.
	if not white_to_move:
		features = features[::-1,:,:]

	# Set the last feature map to be all ones.
	features[:,:,12] = 1

	return features

class TablebaseNetwork:
	NONLINEARITY = [tf.nn.relu]
	FILTERS = 128
	CONV_SIZE = 3
	BLOCK_COUNT = 12
	OUTPUT_CONV_FILTERS = 3
	FC_SIZES = [OUTPUT_CONV_FILTERS * 64, 128, 3]
	FINAL_OUTPUT_SHAPE = [None, 3]

	def __init__(self, scope_name, build_training=False):
		self.scope_name = scope_name
		self.total_parameters = 0
		with tf.variable_scope(scope_name):
			self.build_tower()
		if build_training:
			self.build_training()

	def build_tower(self):
		# Construct input/output placeholders.
		self.input_ph = tf.placeholder(
			tf.float32,
			shape=[None, 8, 8, FEATURE_LAYERS],
			name="input_placeholder")
		self.desired_output_ph = tf.placeholder(
			tf.float32,
			shape=self.FINAL_OUTPUT_SHAPE,
			name="desired_output_placeholder")
		self.learning_rate_ph = tf.placeholder(tf.float32, shape=[], name="learning_rate")
		self.is_training_ph = tf.placeholder(tf.bool, shape=[], name="is_training")
		# Begin constructing the data flow.
		self.parameters = []
		self.flow = self.input_ph
		# Stack an initial convolution.
		self.stack_convolution(self.CONV_SIZE, FEATURE_LAYERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack some number of residual blocks.
		for _ in range(self.BLOCK_COUNT):
			self.stack_block()
		# Stack a final 1x1 convolution transitioning to fully-connected features.
		self.stack_convolution(1, self.FILTERS, self.OUTPUT_CONV_FILTERS, batch_normalization=False)
		# Switch over to fully connected processing by flattening.
		self.flow = tf.reshape(self.flow, [-1, self.FC_SIZES[0]])
		for old_size, new_size in zip(self.FC_SIZES, self.FC_SIZES[1:]):
			self.stack_nonlinearity()
			W = self.new_weight_variable([old_size, new_size])
			b = self.new_bias_variable([new_size])
			self.flow = tf.matmul(self.flow, W) + b
		# The final result of self.flow is now of shape [-1, 3] with logits over (win, draw, loss)

	def build_training(self):
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
		reg_variables = tf.trainable_variables(scope=self.scope_name)
		self.regularization_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
			labels=self.desired_output_ph,
			logits=self.flow,
		))
		self.loss = self.cross_entropy + self.regularization_term

		self.accuracy = tf.reduce_mean(tf.cast(
			tf.equal(tf.argmax(self.flow, 1), tf.argmax(self.desired_output_ph, 1)),
			tf.float32,
		))

		# Associate batch normalization with training.
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope_name)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.MomentumOptimizer(
				learning_rate=self.learning_rate_ph, momentum=0.9).minimize(self.loss)

	def new_weight_variable(self, shape):
		self.total_parameters += np.product(shape)
		# Scale down regular Xavier initialization because we're residual.
		stddev = 0.2 * (2.0 / np.product(shape[:-1]))**0.5
		var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
		self.parameters.append(var)
		return var

	def new_bias_variable(self, shape):
		self.total_parameters += np.product(shape)
		var = tf.Variable(tf.constant(0.1, shape=shape))
		self.parameters.append(var)
		return var

	def stack_convolution(self, kernel_size, old_size, new_size, batch_normalization=True):
		weights = self.new_weight_variable([kernel_size, kernel_size, old_size, new_size])
		self.flow = tf.nn.conv2d(self.flow, weights, strides=[1, 1, 1, 1], padding="SAME")
		if batch_normalization:
			self.flow = tf.layers.batch_normalization(
				self.flow,
				center=True,
				scale=True,
				training=self.is_training_ph)
		else:
			bias = self.new_bias_variable([new_size])
			self.flow = self.flow + bias # TODO: Is += equivalent?

	def stack_nonlinearity(self):
		self.flow = self.NONLINEARITY[0](self.flow)

	def stack_block(self):
		initial_value = self.flow
		# Stack the first convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		self.stack_nonlinearity()
		# Stack the second convolution.
		self.stack_convolution(self.CONV_SIZE, self.FILTERS, self.FILTERS)
		# Add the skip connection.
		self.flow = self.flow + initial_value
		# Stack on the deferred non-linearity.
		self.stack_nonlinearity()

	def train(self, samples, learning_rate):
		self.run_on_samples(self.train_step.run, samples, learning_rate=learning_rate, is_training=True)

	def get_loss(self, samples):
		return self.run_on_samples(self.loss.eval, samples)

	def get_accuracy(self, samples):
		return self.run_on_samples(self.accuracy.eval, samples)

	def run_on_samples(self, f, samples, learning_rate=0.0, is_training=False):
		return f(feed_dict={
			self.input_ph:          samples["features"],
			self.desired_output_ph: samples["outputs"],
			self.learning_rate_ph:  learning_rate,
			self.is_training_ph:    is_training,
		})

# XXX: This is horrifically ugly.
# TODO: Once I have a second change it to not do this horrible graph scraping that breaks if you have other things going on.
def get_batch_norm_vars(net):
	return [
		i for i in tf.global_variables(scope=net.scope_name)
		if "batch_normalization" in i.name and ("moving_mean:0" in i.name or "moving_variance:0" in i.name)
	]

def save_model(net, path):
	x_conv_weights = sess.run(net.parameters)
	x_bn_params = sess.run(get_batch_norm_vars(net))
	np.save(path, [x_conv_weights, x_bn_params])
	print("\x1b[35mSaved model to:\x1b[0m", path)

# XXX: Still horrifically fragile wrt batch norm variables due to the above horrible graph scraping stuff.
def load_model(net, path):
	x_conv_weights, x_bn_params = np.load(path)
	assert len(net.parameters) == len(x_conv_weights), "Parameter count mismatch!"
	operations = []
	for var, value in zip(net.parameters, x_conv_weights):
		operations.append(var.assign(value))
	bn_vars = get_batch_norm_vars(net)
	assert len(bn_vars) == len(x_bn_params), "Bad batch normalization parameter count!"
	for var, value in zip(bn_vars, x_bn_params):
		operations.append(var.assign(value))
	sess.run(operations)

def add_options_to_argparser(parser):
	parser.add_argument("--blocks", metavar="INT", default=8, type=int, help="Number of residual blocks to stack.")
	parser.add_argument("--filters", metavar="INT", default=64, type=int, help="Number of convolutional filters.")
	parser.add_argument("--conv-size", metavar="INT", default=3, type=int, help="Convolution size. e.g. if set to 3 all convolutions are 3x3.")
	parser.add_argument("--final-conv-filters", metavar="INT", default=3, type=int, help="A complicated option. Right before we switch to fully connected processing we reduce the dimensionality of the data out of the convolutional tower. This is the number of filters we reduce to right before flattening into a single vector for fully-connected processing. In AlphaGo Zero this value was 2 for the policy head and 1 for the value head. Here it should probably be at least three, because we are outputting categorical information over three classes.")
	parser.add_argument("--fully-connected-layers", metavar="COMMA-SEPARATED-INTS", default="128,3", type=str, help="Sizes of the fully connected layers you'd like stacked at the end. Must be comma separated values, ending in a 3 because the end of the network is a softmax over (win, draw, loss). You may include as many fully connected layers as you want.")
	parser.add_argument("--nonlinearity", metavar="STR", default="relu", choices=("relu", "leaky-relu", "elu", "sigmoid"), help="What non-linearity to use in the network. Options: relu, leaky-relu, elu, sigmoid")

def set_options_from_args(args):
	# Parse the comma separated fully connected layers string (e.g., "128,64,3") into a list of ints.
	if isinstance(args.fully_connected_layers, str):
		args.fully_connected_layers = list(map(int, args.fully_connected_layers.split(",")))
	TablebaseNetwork.BLOCK_COUNT         = args.blocks
	TablebaseNetwork.FILTERS             = args.filters
	TablebaseNetwork.CONV_SIZE           = args.conv_size
	TablebaseNetwork.OUTPUT_CONV_FILTERS = args.final_conv_filters
	TablebaseNetwork.FC_SIZES            = [args.final_conv_filters * 64] + args.fully_connected_layers
	TablebaseNetwork.NONLINEARITY        = {
		"relu": [tf.nn.relu],
		"leaky-relu": [tf.nn.leaky_relu],
		"elu": [tf.nn.elu],
		"sigmoid": [tf.nn.sigmoid],
	}[args.nonlinearity]

if __name__ == "__main__":
	print("Making a network as a test.")
	nntb = TablebaseNetwork("net/")
	print("Total parameters:", nntb.total_parameters)

