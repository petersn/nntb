#!/usr/bin/python

import os, time, random, itertools
import numpy as np
import tensorflow as tf
import chess, chess.syzygy
import model

MINIBATCH_SIZE = 512
LEARNING_RATE = 0.005
LEARNING_RATE_HALFLIFE = 100e3

tablebase = chess.syzygy.open_tablebases("./syzygy")

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

def save_model():
	global model_save_counter
	model_save_counter += 1
	path = os.path.join("models", "model-%03i.npy" % model_save_counter)
	model.sess = sess
	model.save_model(net, path)

if __name__ == "__main__":
	print "Initializing model."
	net = model.TablebaseNetwork("net/", build_training=True)
	print "Model parameters:", net.total_parameters
	print
	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	print "Generating test sample."
	test_sample = [make_minibatch(1024) for _ in xrange(10)]

	total_steps = 0
	working_time = 0.0
	overall_start_time = time.time()
	print "Beginning training."
	while True:
		lr = LEARNING_RATE * 2**(-total_steps / float(LEARNING_RATE_HALFLIFE))

		loss = np.average([net.get_loss(s) for s in test_sample])
		print "%6i [%s - %s] Loss: %.6f  lr=%.6f" % (
			total_steps,
			to_hms(time.time() - overall_start_time),
			to_hms(working_time),
			loss,
			lr,
		)

		for _ in xrange(200):
			minibatch = make_minibatch(MINIBATCH_SIZE)
			start = time.time()
			net.train(minibatch, lr)
			working_time += time.time() - start
			total_steps += 1

		if total_steps % 10000 == 0:
			save_model()

