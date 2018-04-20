#!/usr/bin/python

from __future__ import print_function
import os, sys, signal, time, random, itertools, argparse
import numpy as np
import tensorflow as tf
import chess, chess.syzygy
import model

piece_combinations = []
for i in range(4):
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
	features, outputs = map(np.array, zip(*[make_training_sample() for _ in range(size)]))
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
	model.add_options_to_argparser(group)

	group = parser.add_argument_group("Training Options", "Options that only affect how training is done.")
	group.add_argument("--syzygy-path", metavar="PATH", required=True, type=str, help="Path to the directory containing all of the Syzygy tablebase files.")
	group.add_argument("--test-sample-thousands", metavar="INT", default=10, type=int, help="Number of thousands of samples to include in the test sample that is used for printing loss information. Setting it higher merely slows down operation, but results in more accurate information.")
	group.add_argument("--learning-rate", metavar="FLOAT", default=0.001, type=float, help="Initial learning rate to use in the learning rate schedule.")
	group.add_argument("--learning-rate-half-life", metavar="FLOAT", default=10e3, type=float, help="Halve the learning rate after this many minibatches (steps).")
	group.add_argument("--minibatch-size", metavar="INT", default=256, type=int, help="Number of training samples in a single minibatch.")
	group.add_argument("--initial-model", metavar="PATH", default=None, type=str, help="Optional path to a previous .npy model file to resume training from. Must have *exactly* the same architecture! There is no checking of this.")
	group.add_argument("--model-output-dir", metavar="PATH", default="models/", type=str, help="Directory in which to dump models as they save. Will dump as model-001.npy, model-002.npy, and so on, overwriting anything that was there before.")
	group.add_argument("--stats-interval", metavar="INT", default=200, type=int, help="Print loss and accuracy every this many minibatches.")
	group.add_argument("--save-interval", metavar="INT", default=10000, type=int, help="Save the model every this many minibatches.")
	group.add_argument("--no-save", action="store_true", help="Disable model saving entirely.")

	args = parser.parse_args()
	print("Got arguments:", args)

	# Configure all of the model hyperparameters from the options we were passed.
	model.set_options_from_args(args)

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
	test_sample = [make_minibatch(1024) for _ in range(args.test_sample_thousands)]

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

	def ctrl_c_handler(signal, frame):
		if args.no_save:
			print("Ctrl+C detected, but not saving model.")
			return
		print("Ctrl+C detected. Saving model.")
		save_model(args)
		sys.exit(0)
	signal.signal(signal.SIGINT, ctrl_c_handler)

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

