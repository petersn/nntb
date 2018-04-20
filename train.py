#!/usr/bin/python

from __future__ import print_function
import os, sys, time, random, itertools, argparse
import numpy as np
import tensorflow as tf
import chess, chess.syzygy
import model

piece_combinations = []
for i in xrange(4):
	piece_combinations.extend(itertools.combinations_with_replacement("pnbrqPNBRQ", i))

RED  = "\x1b[31m"
ENDC = "\x1b[0m"

def make_random_position():
	# Pick random pieces for the two players.
	pieces = ("k", "K") + random.choice(piece_combinations)
	while True:
		board = chess.Board.empty()
		squares = random.sample(chess.SQUARES, len(pieces))
		for square, piece in zip(squares, pieces):
			board.set_piece_at(square, chess.Piece.from_symbol(piece))
		if not board.is_valid():
			continue
		return board

def make_training_sample():
	board = make_random_position()
	result = tablebase.probe_wdl(board)
	features = model.extract_features(board)
	desired_output = (result > 0, result == 0, result < 0)
	return features, desired_output

def make_minibatch(size):
	features, outputs = map(np.array, zip(*[make_training_sample() for _ in xrange(size)]))
	return {"features": features.astype(np.float32), "outputs": outputs.astype(np.float32)}

def to_hms(x):
	x = int(x)
	seconds = x % 60
	minutes = (x // 60) % 60
	hours   = x // 60 // 60
	return "%2i:%02i:%02i" % (hours, minutes, seconds)

model_save_counter = 0

def save_model(args):
	global model_save_counter
	model_save_counter += 1
	path = os.path.join(args.model_output_dir, "model-%03i.npy" % model_save_counter)
	model.save_model(net, path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	group = parser.add_argument_group("Network Options", "Options that affect the network architecture.")
	group.add_argument("--blocks", metavar="INT", default=8, type=int, help="Number of residual blocks to stack.")
	group.add_argument("--filters", metavar="INT", default=64, type=int, help="Number of convolutional filters.")
	group.add_argument("--conv-size", metavar="INT", default=3, type=int, help="Convolution size. e.g. if set to 3 all convolutions are 3x3.")
	group.add_argument("--final-conv-filters", metavar="INT", default=8, type=int, help="A complicated option. Right before we switch to fully connected processing we reduce the dimensionality of the data out of the convolutional tower. This is the number of filters we reduce to right before flattening into a single vector for fully-connected processing. In AlphaGo Zero this value was 2 for the policy head and 1 for the value head. Here it should probably be at least three, because we are outputting categorical information over three classes.")
	group.add_argument("--fully-connected-layers", metavar="COMMA-SEPARATED-INTS", default="128,3", type=str, help="Sizes of the fully connected layers you'd like stacked at the end. Must be comma separated values, ending in a 3 because the end of the network is a softmax over (win, draw, loss). You may include as many fully connected layers as you want.")
	group.add_argument("--nonlinearity", metavar="STR", default="relu", choices=("relu", "leaky-relu", "elu", "sigmoid"), help="What non-linearity to use in the network. Options: relu, leaky-relu, elu, sigmoid")

	group = parser.add_argument_group("Training Options", "Options that only affect how training is done.")
	group.add_argument("--syzygy-path", metavar="PATH", required=True, type=str, help="Path to the directory containing all of the Syzygy tablebase files.")
	group.add_argument("--test-sample-thousands", metavar="INT", default=10, type=int, help="Number of thousands of samples to include in the test sample that is used for printing loss information. Setting it higher merely slows down operation, but results in more accurate information.")
	group.add_argument("--learning-rate", metavar="FLOAT", default=0.01, type=float, help="Initial learning rate to use in the learning rate schedule.")
	group.add_argument("--learning-rate-half-life", metavar="FLOAT", default=100e3, type=float, help="Halve the learning rate after this many minibatches (steps).")
	group.add_argument("--minibatch-size", metavar="INT", default=256, type=int, help="Number of training samples in a single minibatch.")
	group.add_argument("--initial-model", metavar="PATH", default=None, type=str, help="Optional path to a previous .npy model file to resume training from. Must have *exactly* the same architecture! There is no checking of this.")
	group.add_argument("--model-output-dir", metavar="PATH", default="models/", type=str, help="Directory in which to dump models as they save. Will dump as model-001.npy, model-002.npy, and so on, overwriting anything that was there before.")
	group.add_argument("--stats-interval", metavar="INT", default=200, type=int, help="Print loss and accuracy every this many minibatches.")
	group.add_argument("--save-interval", metavar="INT", default=10000, type=int, help="Save the model every this many minibatches.")
	group.add_argument("--no-save", action="store_true", help="Disable model saving entirely.")

	args = parser.parse_args()
	print("Got arguments:", args)
	args.fully_connected_layers = list(map(int, args.fully_connected_layers.split(",")))

	model.TablebaseNetwork.BLOCK_COUNT         = args.blocks
	model.TablebaseNetwork.FILTERS             = args.filters
	model.TablebaseNetwork.CONV_SIZE           = args.conv_size
	model.TablebaseNetwork.OUTPUT_CONV_FILTERS = args.final_conv_filters
	model.TablebaseNetwork.FC_SIZES            = [args.final_conv_filters * 64] + args.fully_connected_layers
	model.TablebaseNetwork.NONLINEARITY        = {
		"relu": [tf.nn.relu],
		"leaky-relu": [tf.nn.leaky_relu],
		"elu": [tf.nn.elu],
		"sigmoid": [tf.nn.sigmoid],
	}[args.nonlinearity]

	# Opening tablebase.
	tablebase = chess.syzygy.open_tablebases(args.syzygy_path)

	print("Initializing model.")
	net = model.TablebaseNetwork("net/", build_training=True)
	print("Model parameters:", net.total_parameters)
	print()
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess

	# If we have a previous model to resume from, resume now.
	if args.initial_model:
		print("Loading model from:", args.initial_model)
		model.load_model(net, args.initial_model)

	print("Generating test sample.")
	# Make sure the test set is chosen deterministically so it's more comparable between runs.
	random.seed(123456789)
	test_sample = [make_minibatch(1024) for _ in xrange(args.test_sample_thousands)]

	def get_test_sample_statistics(test_sample):
		results = {"loss": [], "accuracy": []}
		for s in test_sample:
			loss, acc = sess.run([net.loss, net.accuracy],
				feed_dict={
					net.input_ph:          s["features"],
					net.desired_output_ph: s["outputs"],
					net.is_training_ph:    False,
				})
			results["loss"].append(loss)
			results["accuracy"].append(acc)
		return {k: np.average(v) for k, v in results.items()}

	total_steps = 0
	useful_time = 0.0
	overall_start_time = time.time()
	print("Beginning training.")
	while True:
		lr = args.learning_rate * 2**(-total_steps / (args.learning_rate_half_life))

		if total_steps % args.stats_interval == 0:
			info = get_test_sample_statistics(test_sample)
			print("  Loss: %.6f  Accuracy: %3.5f%%" % (
				info["loss"],
				info["accuracy"] * 100.0,
			))

		if total_steps % args.save_interval == 0 and not args.no_save:
			save_model(args)

		minibatch = make_minibatch(args.minibatch_size)
		start = time.time()
		net.train(minibatch, lr)
		useful_time += time.time() - start
		total_steps += 1
		print("\rSteps: %5i [time: %s - useful time: %s]  lr=%.6f" % (
			total_steps,
			to_hms(time.time() - overall_start_time),
			to_hms(useful_time),
			lr,
		), end="")
		sys.stdout.flush()

