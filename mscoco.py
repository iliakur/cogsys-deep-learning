import sys
import json

import h5py
from blocks.utils import shared_floatx_nans
from blocks.utils import dict_union, dict_subset
from blocks.roles import WEIGHT, add_role

from contrib.imagenet import ImagenetModel
from workspace import *


def imagenet_model_func(matlab_filepath):
    vgg_vd_model = ImagenetModel(matlab_filepath)
    input_var = tensor.tensor4("input")
    output_1k = vgg_vd_model.layers[0].apply(input_var)
    for layer in vgg_vd_model.layers[1:]:
        output_1k = layer.apply(output_1k)
    # operable_model = Model(output_1k)
    # return operable_model.get_theano_function(allow_input_downcast=True)
    return theano.function([input_var], output_1k, allow_input_downcast=True)
    # return vgg_vd_model


class CocoHD5Dataset(Dataset):
    """Very simple interface to the mscoco dataset"""

    def __init__(self, coco_hd5_path, subset=None):
        hd5_file = h5py.File(coco_hd5_path)
        self.images = hd5_file['image']
        self.sequences = hd5_file['sequence']
        if subset:
            self.images = self.images[subset]
            self.sequences = self.sequences[subset]
        self._sources = tuple(hd5_file.keys())
        self.axis_labels = None

    @property
    def num_examples(self):
        return len(self.images)

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError

        return (self.images[request].astype("f"), self.sequences[request].astype("i").T)


def mscoco_stream(dataset, batch_size):
    batch_scheme = SequentialScheme(dataset.num_examples, batch_size=batch_size)
    just_stream = DataStream.default_stream(dataset, iteration_scheme=batch_scheme)
    return just_stream
    # return Mapping(just_stream, transpose_stream)


def mscoco_read_vocab(f_path):
    coco_json = json.load(open(f_path))
    return coco_json['ix_to_word']


class ImageCaptionAttention(AbstractAttention):

    @application(outputs=["attended"], inputs=["attended", 'preprocessed_attended'])
    def take_glimpses(self, attended, preprocessed_attended=None, **kwargs):
        return attended

    def initial_glimpses(self, batch_size, attended):
        return attended


class ContextSimpleRecurrent(SimpleRecurrent):
    """very simple recurrent that's context-aware"""

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None, context=None):
        """Same as SimpleRecurrent.apply except with an additional argument:

        context : :class:`~tensor.TensorVariable`
            Not actually used here, but needed for readout to take it into account
        """
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    def get_dim(self, name):
        if name == "context":
            # this is a dirty hack!
            return self.parents[0].parents[0].readout.merge.input_dims['context']
        return super(ContextSimpleRecurrent, self).get_dim(name)


class ContextRecurrent(SimpleRecurrent):
    """Fully context-aware recurrent brick"""
    # todo:
    # - _allocate and _initialize need to introduce another weight matrix
    # - get dim is now nice
    # - apply only needs to add the context

    def _allocate(self):
        super(ContextRecurrent, self)._allocate()
        R = shared_floatx_nans((1000, self.dim), name="R")
        self.parameters.append(R)
        add_role(self.parameters[2], WEIGHT)

    def _initialize(self):
        self.weights_init.initialize(self.R, self.rng)
        super(ContextRecurrent, self)._allocate()

    @property
    def R(self):
        return self.parameters[2]

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, context, mask=None):
        """Same as SimpleRecurrent.apply except with an additional argument:

        context : :class:`~tensor.TensorVariable`
        """
        next_states = inputs + tensor.dot(states, self.W) + tensor.dot(context, self.R)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    def get_dim(self, name):
        if name == "context":
            # extending dim behavior to "context" name
            return 1000
        return super(ContextRecurrent, self).get_dim(name)


def train_rnn():

    # coco_hd5_path = "/media/data/image_classification/coco.hdf5"
    coco_hd5_path = "/projects/korpora/mscoco/coco.hdf5"
    coco_dataset = CocoHD5Dataset(coco_hd5_path, subset=range(100))
    stream = mscoco_stream(coco_dataset, 5)

    # coco_hd5_path = "/media/data/image_classification/cocotalk.json"
    coco_json_path = '/projects/korpora/mscoco/coco/cocotalk.json'
    coco_vocab = mscoco_read_vocab(coco_json_path)

    # zeros don't correspond to actual words, so we need to make room for one more index
    vocab_size = len(coco_vocab) + 1
    hidden_size = 512

    feedback = LookupFeedback(vocab_size,
                              feedback_dim=vocab_size,
                              name='feedback')
    emitter = SoftmaxEmitter(name="emitter")
    # merger = Merge(input_names=["states", "context"], input_dims={"context": 1000})
    readout = Readout(readout_dim=vocab_size,
                      # source_names=["states", "context"],
                      source_names=["states"],
                      # merge=merger,
                      emitter=emitter,
                      feedback_brick=feedback,
                      name='readout')

    transition = ContextRecurrent(name="transition",
                                  dim=hidden_size,
                                  activation=Tanh())

    generator = SequenceGenerator(readout,
                                  transition,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  name='generator')
    generator.initialize()

    sequences = tensor.imatrix('sequence')
    images = tensor.matrix('image')
    cost = generator.cost(sequences, context=images)
    graph = ComputationGraph(cost)
    # Cost optimization
    optimizer = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=Adam())

    # Monitoring
    # monitor = DataStreamMonitoring(variables=[cost],
    #                                data_stream=stream,
    #                                prefix="mscoco")
    gradient = aggregation.mean(optimizer.total_gradient_norm)
    gradient_monitoring = TrainingDataMonitoring([gradient, cost], every_n_batches=5)

    # Main Loop
    save_path = 'mscoco-rnn-{}-2.tar'.format(hidden_size)
    # save_path = "test-context.tar"
    main_loop = MainLoop(model=Model(cost),
                         data_stream=stream,
                         algorithm=optimizer,
                         extensions=[FinishAfter(after_n_epochs=1),
                                     gradient_monitoring,
                                     Timing(after_epoch=True),
                                     Printing(on_interrupt=True, every_n_batches=5),
                                     Checkpoint(save_path,
                                                every_n_batches=500,
                                                on_interrupt=True,
                                                after_training=True)
                                     # Plot("Example Plot", channels=[['test_cost_simple_xentropy', "test_error_rate"]])
                                     ])
    main_loop.run()


if __name__ == '__main__':
    matlab_filepath = "/projects/korpora/mscoco/coco/imagenet-vgg-verydeep-16.mat"
    cnn_func = imagenet_model_func(matlab_filepath)

    cocotalk_h5_path = "/projects/korpora/mscoco/coco/cocotalk.h5"
    cocotalk_h5 = h5py.File(cocotalk_h5_path)
    test_images = cocotalk_h5['images'][:2]

    print(cnn_func(test_images))
