## Deep Residual Chess Tablebases

The goal of this project is to experiment with how well CNNs can learn Chess endgame tablebases.
Some questions of interest:

* What sort of asymptotic accuracy can one achieve?
* How much model capacity does it take to learn these tablebases? (i.e., what is the asymptotic accuracy as a function of model size?)
* How quickly do we learn them?
* How well do we generalize to a more general chess value function just from endgame training?

### Requirements:

* Python 2 or Python 3 (should hopefully be compatible with both).
* `python-chess` (installable with: `pip install python-chess`)
* TensorFlow (GPU installation is non-trivial. [Follow some guide from Google.](https://www.google.com/search?q=install+gpu+tensorflow) If you're okay with training on the CPU `pip install tensorflow` should suffice.)

Finally, you must have the Syzygy tablebases downloaded somewhere.
You can download them via [this torrent](http://oics.olympuschess.com/tracker/torrents/Syzygy%203-4-5%20Individual%20Files.torrent) (~1 GiB).

### Usage

If you just want to try the default network architecture simply run `train.py` and pass it the path to the Syzygy tablebases:

```
$ python train.py --syzygy-path ~/Downloads/syzygy/
Got arguments: Namespace(blocks=8, conv_size=3, filters=64, final_conv_filters=3, fully_connected_layers='128,3', initial_model=None, learning_rate=0.001, learning_rate_half_life=10000.0, minibatch_size=256, model_output_dir='models/', no_save=False, nonlinearity='relu', save_interval=10000, stats_interval=200, syzygy_path='/home/snp/Downloads/syzygy/', test_sample_thousands=10)
Initializing model.
Model parameters: 622598

Generating test sample.
Beginning training.
  Loss: 1.160830  Accuracy: 25.16602%
Saved model to: models/model-001.npy
Steps:   200 [time:  0:00:24 - useful time:  0:00:09]  lr=0.000986  Loss: 1.142267  Accuracy: 42.92969%
Steps:   400 [time:  0:00:48 - useful time:  0:00:18]  lr=0.000973  Loss: 1.451355  Accuracy: 31.90430%
Steps:   600 [time:  0:01:13 - useful time:  0:00:27]  lr=0.000959  Loss: 1.989292  Accuracy: 35.56641%
Steps:   800 [time:  0:01:37 - useful time:  0:00:36]  lr=0.000946  Loss: 0.522442  Accuracy: 81.17188%
Steps:  1000 [time:  0:02:01 - useful time:  0:00:45]  lr=0.000933  Loss: 0.469151  Accuracy: 81.13281%
Steps:  1200 [time:  0:02:25 - useful time:  0:00:54]  lr=0.000920  Loss: 0.400564  Accuracy: 86.01562%
...
```

Models will be saved every 10,000 steps into `models/`.

Once you have trained a model you can test it out on example positions as follows:

```
$ python predict.py --model-path models/model-002.npy
> k7/8/1Q6/1p2K3/8/5n2/1r6/8 w - - 0 1
Prediction:
  Win:   87.360%
  Draw:  11.863%
  Loss:   0.777%
> k7/8/1Q6/1p2K3/8/5n2/1q6/8 w - - 0 1
Prediction:
  Win:   39.148%
  Draw:  48.989%
  Loss:  11.863%
```

It's valid to feed in positions with more than 5 pieces, even though the network has never seen any such positions in its training.

### Experiment with the architecture!

You can tune all of the various architecture parameters (number of blocks, number of filters, fully connected layers, type of non-linearity, etc.) by passing various options to `train.py`.
To see a full list simply run `train.py --help`.

The current default network architecture is extremely similar to [Leela Chess Zero](https://github.com/glinscott/leela-chess):

* 6 input planes for our pieces, 6 for their pieces, 1 of all ones.
* A batch-normalized convolutional layer to some number of filters to begin the "tower".
* Some number of stacked residual batch-normalized "blocks", identical to those from AlphaZero or Leela Chess Zero.
* A fully connected "win/draw/loss" head, analogous to the value head from AlphaZero or Leela Chess Zero, except ending in a softmax over (win, draw, loss).

To completely change the model architectures, see `model.py`.

