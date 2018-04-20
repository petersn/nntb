## Network Tablebases

The goal of this project is to experiment with how well CNNs can learn Chess endgame tablebases.
Some questions of interest:

* What sort of asymptotic accuracy can one achieve?
* How much model capacity does it take to learn these tablebases? (i.e., what is the asymptotic accuracy as a function of model size?)
* How quickly do we learn them?
* How well do we generalize to a more general chess value function just from endgame training?

### Usage

First you must install the Syzygy tablebases somewhere.
You can download them via [this torrent](http://oics.olympuschess.com/tracker/torrents/Syzygy%203-4-5%20Individual%20Files.torrent).

Once you have Syzygy downloaded somewhere you can train up a network very easily by simply running `train.py` and passing it the path to the Syzygy tablebases:

```
	
```

### Configuration

To experiment with different model architectures, see `model.py`.
The default configuration is extremely similar to [Leela Chess Zero](https://github.com/glinscott/leela-chess):

* 6 input planes for our pieces, 6 for their pieces, 1 of all ones.
* A batch-normalized convolutional layer to some number of filters.
* Some number of stacked residual batch-normalized "blocks", identical to those from AlphaZero or Leela Chess Zero.
* A fully connected "win/draw/loss" head, analogous to the value head from AlphaZero or Leela Chess Zero, except ending in a softmax over (win, draw, loss).

The various parameters (number of blocks, number of filters, fully connected layers, etc.) are configurable by options to `train.py`. To see a full list run:

```
	train.py --help
```

