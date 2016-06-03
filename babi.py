import os
import json
import time

import h5py

from workspace import *
from utils import fav_extensions, default_batch_stream

# ##################
# data-related stuff
# ##################

# Toggle these lines when switching between local runs and on medusa
# DATA_ROOT = "/media/data/babi-tasks-local"
DATA_ROOT = "/home/kurenkov/data"
# Same, but for loading model parameters
MODEL_ROOT = "/home/kurenkov/models"

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

# Vocab size needs to leave space for zero which doesn't correspond to any entry
# We know empirically that it's 19
VOCAB_SIZE = 19
# Following the paper again, see section 4.4
EMBED_DIM = 20
RNG = np.random.RandomState(int(time.time()))


def shared_random(name, shape=(VOCAB_SIZE, EMBED_DIM)):
    # Following the paper, initialized weights with SD of 0.1
    # For some reason numpy calls SD "scale"
    # gotta make sure it's float32
    randomness = RNG.normal(scale=0.1, size=shape).astype('f')
    return theano.shared(randomness, name=name)


def fake3d_shared_random(name, shape=None):
    if not shape:
        shape = (1, VOCAB_SIZE, EMBED_DIM)
    return shared_random(name, shape=shape)


def one_hot_sequence(indices_tensor_sequence):
    # unfortunately theano currently only supports vectors as input, so we have
    # to use scan for this.
    # the result is a 3D tensor
    return theano.map(tensor.extra_ops.to_one_hot,
                      sequences=[indices_tensor_sequence],
                      non_sequences=[VOCAB_SIZE])[0]


def one_hot_items(indeces_2d):
    one_hot_items_3d = one_hot_sequence(indeces_2d)
    # each item is an array of word 1-hot vectors
    # we sum over the second dimension to get a sentences by embedding matrix
    return one_hot_items_3d.sum(axis=1)


def mapped_dot(vectors, item):
    return theano.map(tensor.dot, sequences=[vectors], non_sequences=[item])


def repeat_batched_dot(vectors, item):
    repeated = item.repeat(vectors.shape[0], axis=0)
    return tensor.batched_dot(vectors, repeated)


def flat_softmax(prob_tensor):
    return tensor.nnet.softmax(prob_tensor).flatten()


def n2n_memory_layer(x_set, u, A, C):

    # Embeddings
    m_set = mapped_dot(x_set, A)[0]
    c_set = mapped_dot(x_set, C)[0]

    # Memory weights
    p = flat_softmax(mapped_dot(m_set, u)[0])

    # Output ("o" in the paper)
    o = p.dot(c_set)

    return o + u


def n2n_network(x_bch, q_bch, A, B, C, W):

    # Inputs converted to one-hot representations
    x_one_hot_bch = theano.map(one_hot_items, sequences=[x_bch])[0]
    q_one_hot_bch = one_hot_items(q_bch)

    u_bch = repeat_batched_dot(q_one_hot_bch, B)

    o_bch = theano.map(n2n_memory_layer,
                       sequences=[x_one_hot_bch, u_bch],
                       non_sequences=[A, C])[0]

    # Answer
    a_hat = tensor.nnet.softmax(repeat_batched_dot(o_bch, W))

    return a_hat


def main(mode):

    # raw input batches coming in
    x = tensor.ltensor3('stories')
    q = tensor.lmatrix('questions')
    a = tensor.lvector('answers')

    if mode == "train":

        # Embedding weights for one layer
        A = shared_random('A')
        B = fake3d_shared_random('B')
        C = shared_random('C')
        W = fake3d_shared_random('W', shape=(1, EMBED_DIM, VOCAB_SIZE))

        # getting an estimate
        a_hat = n2n_network(x, q, A, B, C, W)

        # Improving answer estimate
        batch_cost = tensor.nnet.categorical_crossentropy(a_hat, a).mean()
        batch_cost.name = "cc-entropy average"

        # TODO:
        # - implement gradient clipping
        # - the step rule they had
        optimizer = GradientDescent(cost=batch_cost,
                                    parameters=[A, B, C, W],
                                    step_rule=Scale(learning_rate=0.01))
        gradient_norm = aggregation.mean(optimizer.total_gradient_norm)

        # Feed actual data
        babi_ds = BaBiDataset(os.path.join(DATA_ROOT, "babi-task2-300stories.h5"))
        babi_stream = default_batch_stream(babi_ds, 32)

        # train for 60 epochs, monitor cost and gradient norm, write to file
        loop_extensions = fav_extensions(60, [batch_cost, gradient_norm],
                                         "babi-task2-60-epochs.tar", monitor_freq=50)
        main_loop = MainLoop(algorithm=optimizer,
                             extensions=loop_extensions,
                             data_stream=babi_stream)
        main_loop.run()

    elif mode == 'test':
        # to-do: load paramdict
        model_fname = "babi-task2-60-epochs.tar"
        param_dict = blocksIO.load_parameters(os.path.join(MODEL_ROOT, model_fname))

        # Embedding weights for one layer
        A = param_dict('/A')
        B = param_dict('/B')
        C = param_dict('/C')
        W = param_dict('/W')

        a_hat = n2n_network(x, q, A, B, C, W)

        qa_solver = theano.function([x, q], outputs=a_hat)

        test_data_path = os.path.join(DATA_ROOT, "babi-task2-200stories-test.h5")
        with h5py.File(test_data_path) as test_data_h5:
            stories = test_data_h5['stories']
            questions = test_data_h5['questions']
            answer_prob_dists = qa_solver(stories, questions)
            np.save('test_data_answers', answer_prob_dists)

    # Return answer index
    # return a_hat_bch.argmax()

if __name__ == '__main__':
    main("test")
