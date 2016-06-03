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


class LayerParams(object):
    """class to wrap around layer parameters"""

    def __init__(self, level, A=None, C=None):
        self.level = level
        self.set_param("A", A)
        self.set_param("C", C)

    def set_param(self, name, value):
        shared_var_name = name + str(self.level)
        if value is None:
            setattr(self, name, shared_random(shared_var_name))
        else:
            setattr(self, name, value)
            getattr(self, name).name = shared_var_name


def n2n_memory_layer(x_set, u, A, C):

    # Embeddings
    m_set = mapped_dot(x_set, A)[0]
    c_set = mapped_dot(x_set, C)[0]

    # Memory weights
    p = flat_softmax(mapped_dot(m_set, u)[0])

    # Output ("o" in the paper)
    o = p.dot(c_set)

    return o + u


def n2n_network(x_bch, q_bch, layers, B, W):

    # Inputs converted to one-hot representations
    x_one_hot_bch = theano.map(one_hot_items, sequences=[x_bch])[0]
    q_one_hot_bch = one_hot_items(q_bch)

    u_bch = repeat_batched_dot(q_one_hot_bch, B)

    for layer in layers:
        parameters = [layer.A, layer.C]
        u_bch = theano.map(n2n_memory_layer,
                           sequences=[x_one_hot_bch, u_bch],
                           non_sequences=parameters)[0]

    o_bch = u_bch
    # Answer
    a_hat = tensor.nnet.softmax(repeat_batched_dot(o_bch, W))

    return a_hat


def main(mode):
    model_fname = "babi-task2-60-epochs-3-layers-rnn.tar"
    model_save_path = os.path.join(MODEL_ROOT, model_fname)

    # raw input batches coming in
    x = tensor.ltensor3('stories')
    q = tensor.lmatrix('questions')
    a = tensor.lvector('answers')

    if mode == "train":

        # Weights for questions and final answer
        B = fake3d_shared_random('B')
        W = fake3d_shared_random('W', shape=(1, EMBED_DIM, VOCAB_SIZE))

        # per layer embedding weights
        shared_layer = LayerParams("")
        layers = [
            shared_layer,
            shared_layer,
            shared_layer
        ]
        # getting a network estimate
        a_hat = n2n_network(x, q, layers, B, W)

        # Improving network estimate
        batch_cost = tensor.nnet.categorical_crossentropy(a_hat, a).mean()
        batch_cost.name = "cc-entropy average"

        # TODO:
        # - implement gradient clipping
        # - the step rule they had
        relevant_params = [B, W, shared_layer.A, shared_layer.C]
        # In case no weights are shared
        # for layer in layers:
        #     relevant_params.append(layer.A)
        #     relevant_params.append(layer.C)

        optimizer = GradientDescent(cost=batch_cost,
                                    parameters=relevant_params,
                                    # step_rule=Scale(learning_rate=0.01)
                                    step_rule=Adam()
                                    )
        gradient_norm = aggregation.mean(optimizer.total_gradient_norm)

        # Feed actual data
        babi_ds = BaBiDataset(os.path.join(DATA_ROOT, "babi-task2-300stories.h5"))
        babi_stream = default_batch_stream(babi_ds, 32)

        # train for 60 epochs, monitor cost and gradient norm, write to file
        loop_extensions = fav_extensions(60,
                                         model_save_path,
                                         [batch_cost, gradient_norm],
                                         every_n_batches=50)
        main_loop = MainLoop(algorithm=optimizer,
                             extensions=loop_extensions,
                             data_stream=babi_stream)
        main_loop.run()

    elif mode == 'test':
        # to-do: load paramdict
        # model_fname = "babi-task2-60-epochs-3-layers-rnn.tar"
        param_dict = blocksIO.load_parameter_values(model_save_path)

        # Embedding weights for one layer
        B = theano.shared(param_dict['/B'], "B")
        W = theano.shared(param_dict['/W'], "W")

        shared_layer = LayerParams("",
                                   A=theano.shared(param_dict['/A']),
                                   C=theano.shared(param_dict['/C']))
        # layers = [
        #     LayerParams(1,
        #                 A=theano.shared(param_dict['/A1']),
        #                 C=theano.shared(param_dict['/C1'])),
        #     LayerParams(2,
        #                 A=theano.shared(param_dict['/A2']),
        #                 C=theano.shared(param_dict['/C2'])),
        #     LayerParams(3,
        #                 A=theano.shared(param_dict['/A3']),
        #                 C=theano.shared(param_dict['/C3'])),
        # ]
        layers = [
            shared_layer,
            shared_layer,
            shared_layer,
        ]

        a_hat = n2n_network(x, q, layers, B, W)

        qa_solver = theano.function([x, q], outputs=a_hat)

        # test_fname = "babi-task2-200stories-test.h5"
        test_fname = "babi-task2-300stories.h5"
        test_data_path = os.path.join(DATA_ROOT, test_fname)
        with h5py.File(test_data_path) as test_data_h5:
            stories = test_data_h5['stories']
            questions = test_data_h5['questions']
            answer_prob_dists = qa_solver(stories, questions)
            # For further data munging save the probabilites
            save_to_fname = 'test_on_train_answers'
            print("Saving answer prob-dists to: {}".format(save_to_fname))
            np.save(save_to_fname, answer_prob_dists)
            # For evaluation print the percent correct
            correct_answers = np.array(test_data_h5['answers'])
            network_answers = answer_prob_dists.argmax(axis=1)
            # mean of bool vector is its sum / its len
            percent_correct = (correct_answers == network_answers).mean()
            print("Percent correct on train data: {}%".format(percent_correct * 100))

if __name__ == '__main__':
    main("test")
