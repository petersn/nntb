#!/usr/bin/python

from __future__ import print_function
import readline
import argparse
import numpy as np
import tensorflow as tf
import chess
import model

# Python2/3 compatibility.
try:
	input = raw_input
except NameError:
	pass

def softmax(logits):
	"""Somewhat numerically stable softmax routine."""
	e_x = np.exp(logits - np.max(logits))
	return e_x / e_x.sum()

def predict(fen):
	board = chess.Board(fen)
	features = model.extract_features(board)
	logits, = sess.run(net.flow, feed_dict={
		net.input_ph: [features],
		net.is_training_ph: False,
	})
	assert logits.shape == (3,)
	win, draw, loss = softmax(logits)
	return win, draw, loss

def print_probs(win, draw, loss):
	print("Prediction:\n  Win:  %7.3f%%\n  Draw: %7.3f%%\n  Loss: %7.3f%%" % (100 * win, 100 * draw, 100 * loss))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	group = parser.add_argument_group("Network Options", "Options that affect the network architecture.")
	model.add_options_to_argparser(group)

	parser.add_argument("--model-path", metavar="PATH", required=True, type=str, help="Path to the .npy file with the current model.")
	parser.add_argument("--fen", metavar="FEN", default=None, type=str, help="FEN string of input position to predict.")
	parser.add_argument("--interactive", action="store_true", help="If passed then enter an interactive prediction session.")
	args = parser.parse_args()

	# Configure all of the model hyperparameters from the options we were passed.
	model.set_options_from_args(args)

	net = model.TablebaseNetwork("net/", build_training=True)
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	model.sess = sess
	model.load_model(net, args.model_path)

	if args.fen != None:
		win, draw, loss = predict(args.fen)
		print_probs(win, draw, loss)

	if args.interactive:
		while True:
			fen = input("> ")
			try:
				win, draw, loss = predict(fen)
				print_probs(win, draw, loss)
			except Exception as e:
				print("Error:", e)

