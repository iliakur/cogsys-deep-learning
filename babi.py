import os
import json

import h5py

from workspace import *
from utils import fav_extensions, default_batch_stream

# data-related stuff
# Toggle these lines when switching between local runs and on medusa
# DATA_ROOT = "/media/data/babi-tasks-local"
DATA_ROOT = "/home/kurenkov/data"


def babi_vocab(f_path="babi-task2-300stories.vocab.json"):
    with open(os.path.join(DATA_ROOT, f_path)) as vocab_file:
        return json.load(vocab_file)


class BaBiDataset(Dataset):
    """Very simple interface to the mscoco dataset"""

    def __init__(self, hd5_path, subset=None):
        self.f = h5py.File(hd5_path)
        self._sources = tuple(self.f.keys())
        self.axis_labels = None

    @property
    def num_examples(self):
        return len(self.f[self.sources[0]])

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError

        return tuple(self.f[src][request] for src in self._sources)

# ##################
# Defining the Model
# ##################

vocab = babi_vocab()

# Vocab size needs to leave space for zero which doesn't correspond to any entry
VOCAB_SIZE = len(vocab) + 1
# Following the paper again, see section 4.4
EMBED_DIM = 20
RNG = np.random.RandomState(1984)


def shared_random(name, shape=(VOCAB_SIZE, EMBED_DIM)):
    # Following the paper, initialized weights with SD of 0.1
    # For some reason numpy calls SD "scale"
    # gotta make sure it's float32
    randomness = RNG.normal(scale=0.1, size=shape).astype('f')
    return theano.shared(randomness, name=name)


def mapped_dot(vectors, item):
    return theano.map(tensor.dot, sequences=[vectors], non_sequences=[item])


def one_hot_sum(indices_tensor):
    one_hot = tensor.extra_ops.to_one_hot(indices_tensor, VOCAB_SIZE)
    # sum along first axis
    return one_hot.sum(axis=0)


def flat_softmax(prob_tensor):
    return tensor.nnet.softmax(prob_tensor).flatten()


# Embedding weights for one layer
# TODO: make them play nicely with scan
A = shared_random('A')
B = shared_random('B')
C = shared_random('C')


def n2n_memory_layer(x_indeces, q_indeces):

    # Inputs prepared for embedding
    x_set = theano.map(one_hot_sum, sequences=[x_indeces])[0]
    q = one_hot_sum(q_indeces)

    # Embeddings
    m_set = mapped_dot(x_set, A)[0]
    c_set = mapped_dot(x_set, C)[0]
    u = q.dot(B)

    # Memory weights
    p = flat_softmax(mapped_dot(m_set, u)[0])

    # Output ("o" in the paper)
    o = p.dot(c_set)

    return o + u


def train_n2n():
    x_batch = tensor.ltensor3('stories')
    q_batch = tensor.lmatrix('questions')
    o_batch = theano.map(n2n_memory_layer, sequences=[x_batch, q_batch])[0]

    # Answer
    W = shared_random('W', shape=(EMBED_DIM, VOCAB_SIZE))
    a_hat = tensor.nnet.softmax(mapped_dot(o_batch, W)[0])

    # Improving answer estimate
    a = tensor.lvector('answers')
    batch_cost = tensor.nnet.categorical_crossentropy(a_hat, a).mean()

    # TODO:
    # - implement gradient clipping
    # - the step rule they had
    # gradient_clipper = algorithms.StepClipping
    optimizer = GradientDescent(cost=batch_cost,
                                parameters=[A, B, C, W],
                                step_rule=Scale(learning_rate=0.01))
    gradient_norm = aggregation.mean(optimizer.total_gradient_norm)

    # Feed actual data
    babi_ds = BaBiDataset(os.path.join(DATA_ROOT, "babi-task2-300stories.h5"))
    babi_stream = default_batch_stream(babi_ds, 32)

    # train for 20 epochs, monitor cost and gradient norm, write to file
    loop_extensions = fav_extensions(20, [batch_cost, gradient_norm],
                                     "babi-task2.tar", monitor_freq=32)
    main_loop = MainLoop(algorithm=optimizer,
                         extensions=loop_extensions,
                         data_stream=babi_stream)
    main_loop.run()

if __name__ == '__main__':
    train_n2n()
