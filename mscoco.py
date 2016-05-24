import sys
import json

import h5py
from blocks.utils import shared_floatx_nans
from blocks.utils import dict_union, dict_subset
from blocks.roles import WEIGHT, add_role

from contrib.imagenet import ImagenetModel
from workspace import *
from utils import transpose_stream


def imagenet_model_func(matlab_filepath):
    vgg_vd_model = ImagenetModel(matlab_filepath)
    input_var = tensor.tensor4("input")
    output_1k = Sequence(br.apply for br in vgg_vd_model.layers).apply(input_var)
    operable_model = Model(output_1k)
    return operable_model.get_top_bricks()
    # return theano.function([input_var], output_1k)
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

        return (self.images[request], self.sequences[request].astype("i").T)


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


class ImageContextRecurrent(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, **kwargs):
        self.dim = dim
        # kwargs.setdefault('children', []).extend(children)
        # children = [] + children
        super(ImageContextRecurrent, self).__init__(**kwargs)
        # super().__init__(children=children, **kwargs)
        # This is a dirty hack, dunno why children cannot be passed as a kwarg
        self.children = [activation] + kwargs.get('children', [])

    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (ImageContextRecurrent.apply.sequences +
                    ImageContextRecurrent.apply.states):
            return self.dim
        return super(ImageContextRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim), name="W"))
        add_role(self.parameters[0], WEIGHT)

        # NB no parameters for initial state

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None, **kwargs):
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(contexts=["context"])
    def initial_states(self, batch_size, *args, **kwargs):
        init = kwargs["context"]
        return init.T

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.apply.states


class ContextSimpleRecurrent(SimpleRecurrent):
    """very simple recurrent that's context-aware"""

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=['context'])
    def apply(self, inputs, states, mask=None):
        """Apply the simple transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        next_states = inputs + tensor.dot(states, self.W)
        next_states = self.children[0].apply(next_states)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    def get_dim(self, name):
        if name == "context":
            return self.merge.input_dims['context']
        return super(ContextSimpleRecurrent, self).get_dim(name)


class ContextReadout(Readout):
    """Context-aware Readout"""

    def get_dim(self, name):
        if name == "context":
            return self.merge.input_dims['context']
        return super(ContextReadout, self).get_dim(name)


def train_rnn():

    # coco_hd5_path = "/media/data/image_classification/coco.hdf5"
    coco_hd5_path = "/projects/korpora/mscoco/coco.hdf5"
    coco_dataset = CocoHD5Dataset(coco_hd5_path, subset=range(10))
    stream = mscoco_stream(coco_dataset, 50)

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
    merger = Merge(input_names=["states", "context"], input_dims={"context": 1000})
    readout = ContextReadout(readout_dim=vocab_size,
                             source_names=["states", "context"],
                             merge=merger,
                             emitter=emitter,
                             feedback_brick=feedback,
                             name='readout')

    transition = ContextSimpleRecurrent(name="transition",
                                        dim=hidden_size,
                                        activation=Rectifier())

    generator = SequenceGenerator(readout,
                                  transition,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  name='generator')
    generator.initialize()

    sequences = tensor.imatrix('sequence')
    images = tensor.dmatrix('image')
    cost = generator.cost(sequences, context=images)
    graph = ComputationGraph(cost)
    # Cost optimization
    optimizer = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=Adam())

    # Monitoring
    monitor = DataStreamMonitoring(variables=[cost],
                                   data_stream=stream,
                                   prefix="mscoco")

    # Main Loop
    save_path = 'mscoco-rnn-{}.tar'.format(hidden_size)
    main_loop = MainLoop(model=Model(cost),
                         data_stream=stream,
                         algorithm=optimizer,
                         extensions=[
                             monitor,
                             FinishAfter(after_n_epochs=2),
                             Printing(on_interrupt=True),
                             Timing(on_interrupt=True),
                             Checkpoint(save_path,
                                        every_n_batches=500,
                                        on_interrupt=True)
                             # Plot("Example Plot", channels=[['test_cost_simple_xentropy', "test_error_rate"]])
    ])
    main_loop.run()


if __name__ == '__main__':
    # matlab_filepath = sys.argv[1]
    # print(imagenet_model_func(matlab_filepath))
    train_rnn()
