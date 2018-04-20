## Network Tablebases

The goal of this project is to experiment with how well CNNs can learn Chess endgame tablebases.
Some questions of interest:

* What sort of asymptotic accuracy can one achieve?
* How much model capacity does it take to learn these tablebases? (i.e., what is the asymptotic accuracy as a function of model size?)
* How quickly do we learn them?
* How well do we generalize to a more general chess value function just from endgame training?

### Requirements:

Requirements:

* Python 2 or Python 3 (should hopefully be compatible with both).
* `python-chess` (installable with: `pip install python-chess`)
* TensorFlow (GPU installation is non-trivial. [Follow some guide from Google.](https://www.google.com/search?q=install+gpu+tensorflow) If you're okay with training on the CPU `pip install tensorflow` should suffice.)

Finally, you must have the Syzygy tablebases downloaded somewhere.
You can download them via [this torrent](http://oics.olympuschess.com/tracker/torrents/Syzygy%203-4-5%20Individual%20Files.torrent) (~1 GiB).

### Usage

If you just want to try the default network architecture simply run `train.py` and passit the path to the Syzygy tablebases:

```
	
```

Training time is dominated by generating the training samples (which requires probing the tablebase hundreds of times per minibatch), so GPU speed shouldn't matter that much.

### Experiment with the architecture!

The default configuration is extremely similar to [Leela Chess Zero](https://github.com/glinscott/leela-chess):

* 6 input planes for our pieces, 6 for their pieces, 1 of all ones.
* A batch-normalized convolutional layer to some number of filters.
* Some number of stacked residual batch-normalized "blocks", identical to those from AlphaZero or Leela Chess Zero.
* A fully connected "win/draw/loss" head, analogous to the value head from AlphaZero or Leela Chess Zero, except ending in a softmax over (win, draw, loss).

The various parameters (number of blocks, number of filters, fully connected layers, type of non-linearity, etc.) are configurable by options to `train.py`. To see a full list run:

```
	train.py --help
```

To completely change the model architectures, see `model.py`.

