#!/usr/bin/python

from __future__ import print_function
import os, sys, signal, time, random, itertools, argparse, threading, queue, multiprocessing
import numpy as np
import tensorflow as tf
import chess, chess.syzygy
from tqdm import tqdm
import model

material = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}

# Generate all piece combinations of up to three non-king pieces.
# Later to use 6-piece tablebases one could increase this 4 to a 5.
main_piece_combinations = []
short_piece_combinations = []
for i in range(1, 5):
    L = list(itertools.combinations_with_replacement("pnbrqPNBRQ", i))
    print("Piece combinations with:", i, "count:", len(L))
    if i == 4:
        main_piece_combinations.extend(L)
    else:
        short_piece_combinations.extend(L)

combos_by_material_difference = {}
for combo in main_piece_combinations:
    material_difference = abs(sum(material[c.lower()] * {True: +1, False: -1}[c == c.upper()] for c in combo))
    if material_difference not in combos_by_material_difference:
        combos_by_material_difference[material_difference] = []
    combos_by_material_difference[material_difference].append(combo)

print("Main piece combinations:", len(main_piece_combinations))
print("Short piece combinations:", len(short_piece_combinations))
very_balanced = []
medium_balanced = []
highly_imbalanced = []
for md in sorted(combos_by_material_difference.keys()):
    positions = combos_by_material_difference[md]
    print("Positions with material difference %2i: %i" % (md, len(positions)))
    if md <= 2:
        very_balanced.extend(positions)
    elif md <= 5:
        medium_balanced.extend(positions)
    else:
        highly_imbalanced.extend(positions)

print("Balance bucket sizes: %i - %i - %i (short: %i)" % (
    len(very_balanced), len(medium_balanced), len(highly_imbalanced), len(short_piece_combinations)
))

def make_random_position():
    """make_random_position() -> random legal chess.Board with at most five pieces"""
    # Pick random pieces for the two players, and add in kings.
    r = random.random()
    if r < 0.01:
        extra_pieces = random.choice(short_piece_combinations)
    elif r < 0.03:
        extra_pieces = random.choice(highly_imbalanced)
    elif r <= 0.15:
        extra_pieces = random.choice(medium_balanced)
    else:
        extra_pieces = random.choice(very_balanced)
    pieces = ("k", "K") + extra_pieces
    while True:
        board = chess.Board.empty()
        squares = random.sample(chess.SQUARES, len(pieces))
        for square, piece in zip(squares, pieces):
            board.set_piece_at(square, chess.Piece.from_symbol(piece))
        if not board.is_valid():
            continue
        return board

while False:
    s = str(make_random_position())
    s = "".join(
        "\x1b[35m" + c + "\x1b[0m" if c.isupper() else c
        for c in s
    )
    print(s)
    input(">")

def make_training_sample():
    """make_training_sample() -> (np.array of shape (8, 8, 13), (win bool, draw bool, loss bool))
    Returns features and 3-tuple of bools of what the outcome is from a randomly selected board position.
    You must first set the global variable `tablebase` to use this!
    For example: train.tablebase = chess.syzygy.open_tablebases("./syzygy/")
    """
    board = make_random_position()
    result = tablebase.probe_wdl(board)
    features = model.extract_features(board)
    desired_output = (result > 0, result == 0, result < 0)
    return features, desired_output

def make_minibatch(size):
    """make_minibatch(size) -> ...
    returns a dict of the form: {
        "features": np.array of shape (size, 8, 8, 13),
        "outputs":  np.array of shape(size, 3),
    }
    See `make_training_sample` for info on setting the global variable `tablebase` so you can call this.
    """
    features, outputs = map(np.array, zip(*[make_training_sample() for _ in range(size)]))
    return {"features": features.astype(np.float32), "outputs": outputs.astype(np.float32)}

def to_hms(x):
    """to_hms(x) -> nicely formatted hours:minutes:seconds from a floating point time delta"""
    x = int(x)
    seconds = x % 60
    minutes = (x // 60) % 60
    hours   = x // 60 // 60
    return "%2i:%02i:%02i" % (hours, minutes, seconds)

model_save_counter = 0

def save_model(args):
    """save_model(args: argparse.Namespace)
    Increments the save counter, and saves the global model.
    You must first set `model.sess`. See `model.save_model` for more info.
    """
    global model_save_counter
    model_save_counter += 1
    path = os.path.join(args.model_output_dir, "model-%03i.npy" % model_save_counter)
    model.save_model(net, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add the various network architecture hyperparameter options into our argument parser.
    group = parser.add_argument_group("Network Options", "Options that affect the network architecture.")
    model.add_options_to_argparser(group)

    # Add all of our training options.
    group = parser.add_argument_group("Training Options", "Options that only affect how training is done.")
    group.add_argument("--syzygy-path", metavar="PATH", required=True, type=str, help="Path to the directory containing all of the Syzygy tablebase files.")
    group.add_argument("--test-sample-thousands", metavar="INT", default=10, type=int, help="Number of thousands of samples to include in the test sample that is used for printing loss information. Setting it higher merely slows down operation, but results in more accurate information.")
    group.add_argument("--learning-rate", metavar="FLOAT", default=0.01, type=float, help="Initial learning rate to use in the learning rate schedule.")
    group.add_argument("--learning-rate-half-life", metavar="FLOAT", default=1e6, type=float, help="Halve the learning rate after this many minibatches (steps).")
    group.add_argument("--warm-up-steps", metavar="INT", default=10000, type=int, help="Steps over which to linearly ramp up learning rate.")
    group.add_argument("--minibatch-size", metavar="INT", default=512, type=int, help="Number of training samples in a single minibatch.")
    group.add_argument("--initial-model", metavar="PATH", default=None, type=str, help="Optional path to a previous .npy model file to resume training from. Must have *exactly* the same architecture! There is no checking of this.")
    group.add_argument("--model-output-dir", metavar="PATH", default="models/", type=str, help="Directory in which to dump models as they save. Will dump as model-001.npy, model-002.npy, and so on, overwriting anything that was there before.")
    group.add_argument("--stats-interval", metavar="INT", default=2000, type=int, help="Print loss and accuracy every this many minibatches.")
    group.add_argument("--save-interval", metavar="INT", default=100000, type=int, help="Save the model every this many minibatches.")
    group.add_argument("--no-save", action="store_true", help="Disable model saving entirely.")
    group.add_argument("--test-minibatch-generation", action="store_true", help="Test the speed of minibatch generation, to determine if it's the bottleneck in training.")

    args = parser.parse_args()
    print("Got arguments:", args)

    # Configure all of the model hyperparameters from the options we were passed.
    # This is where the number of blocks, filters, etc. in the network gets set.
    model.set_options_from_args(args)

    # Open the tablebases.
    tablebase = chess.syzygy.Tablebase()
    tablebase.add_directory(args.syzygy_path)

#    from tqdm import tqdm
#    for _ in tqdm(range(1000)):
#        make_minibatch(256)

    # Make a network.
    print("Initializing model.")
    net = model.TablebaseNetwork("net/", build_training=True)
    print("Model parameters:", net.total_parameters)
    print()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # We must pass our session into `model` in order for `model.save_model` and `model.load_model` to have access to it.
    model.sess = sess

    # If we have a previous model to resume from, resume now.
    if args.initial_model:
        print("Loading model from:", args.initial_model)
        model.load_model(net, args.initial_model)

    print("Generating test sample.")
    # Make sure the test set is chosen deterministically so it's more comparable between runs.
    random.seed(123456789)
    test_sample = [make_minibatch(1024) for _ in tqdm(range(args.test_sample_thousands))]

    def get_test_sample_statistics(test_sample):
        """get_test_sample_statistics(test_sample: list of minibatch) -> dict of info
        Returns a dict of the form {"loss": float, "accuracy": float}
        Takes as input a list of minibatches, as output by `make_minibatch`.

        The sole purpose of taking in a list of minibatches is to avoid calling
        Tensorflow on a single massive minibatch, which unfortunately currently has
        the behavior of allocating massive tensors, which might run us out of VRAM.
        """
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

    # Make ctrl + c save the model out right before we exit.
    def ctrl_c_handler(signal, frame):
        if args.no_save:
            print(" Ctrl+C detected, but not saving model.")
            return
        print(" Ctrl+C detected. Saving model.")
        save_model(args)
        sys.exit(0)
    signal.signal(signal.SIGINT, ctrl_c_handler)

    USE_THREADS = True
    if USE_THREADS:
        minibatch_queue = queue.Queue()
        def worker_thread():
            print("Starting worker thread.")
            while True:
                if minibatch_queue.qsize() > 100:
                    print("Sleeping.")
                    time.sleep(0.5)
                minibatch = make_minibatch(args.minibatch_size)
                minibatch_queue.put(minibatch)
        worker_thread = threading.Thread(target=worker_thread)
        worker_thread.start()
    else:
        minibatch_queue = multiprocessing.Queue()
        def worker_process(seed, m_queue, size):
            random.seed(seed)
            while True:
                if m_queue.qsize() > 100:
                    time.sleep(0.1)
                minibatch = make_minibatch(size)
                m_queue.put(minibatch)
        processes = []
        for _ in range(4):
            p = multiprocessing.Process(
                target=worker_process,
                args=(
                    random.getrandbits(64), minibatch_queue, args.minibatch_size
                ),
            )
            p.daemon = True
            p.start()
            processes.append(p)

    if args.test_minibatch_generation:
        from tqdm import tqdm
        for _ in tqdm(range(1000)):
            make_minibatch(1)
        for _ in tqdm(range(1000)):
            minibatch_queue.get()

    running_in_sample_loss = 3.0
    total_steps = 0
    useful_time = 0.0
    overall_start_time = time.time()
    print("Beginning training.")

    while True:
        # Compute our current learning rate based on our exponentially decaying schedule.
        lr = args.learning_rate * 2**(-total_steps / (args.learning_rate_half_life))

        # Implement a warm-up period.
        if total_steps < args.warm_up_steps:
            lr *= (1 + total_steps) / args.warm_up_steps

        # Periodically print out loss.
        if total_steps % args.stats_interval == 0:
            info = get_test_sample_statistics(test_sample)
            print("  Loss: %.6f  Accuracy: %3.5f%%" % (
                info["loss"],
                info["accuracy"] * 100.0,
            ))

        # Periodically save the model.
        if total_steps % args.save_interval == 0 and not args.no_save:
            save_model(args)

        # Generate a single minibatch, and train on it.
        minibatch = minibatch_queue.get()
        start = time.time()
        in_sample_loss, _ = sess.run(
            (net.cross_entropy, net.train_step),
            feed_dict={
                net.input_ph:          minibatch["features"],
                net.desired_output_ph: minibatch["outputs"],
                net.learning_rate_ph:  lr,
                net.is_training_ph:    True,
            },
        )
        useful_time += time.time() - start
        ALPHA = 1 / 1000.0
        running_in_sample_loss *= 1 - ALPHA
        running_in_sample_loss += ALPHA * in_sample_loss
        #net.train(minibatch, lr)
        total_steps += 1

        # Update status line.
        print("\rSteps: %5i [time: %s - useful time: %s]  lr=%.6f  l=%.6f" % (
            total_steps,
            to_hms(time.time() - overall_start_time),
            to_hms(useful_time),
            lr,
            running_in_sample_loss,
        ), end="")
        sys.stdout.flush()

