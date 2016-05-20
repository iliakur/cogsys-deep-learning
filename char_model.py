from itertools import chain
from collections import Counter

from nltk.corpus import brown

from workspace import *
from utils import transpose_stream

logger = logging.getLogger('char_model')
logger.addHandler(stream_log_handler)


def build_word_id_dict(word_seq, cutoff=5):
    counts = Counter(word_seq)
    frequent_enough = (word for word in counts if counts[word] >= cutoff)
    return dict((word, w_id + 1) for w_id, word in enumerate(frequent_enough))


def convert_to_ids(word_seq, word_id_dict):
    return np.array([word_id_dict[word] for word in word_seq])


def words_to_ids(vocab):
    def converter(word_seq):
        return convert_to_ids(word_seq, vocab)
    return converter


def nltk_corpus_ids(nltk_corpus):
    corpus_sents = [" ".join(sent) for sent in nltk_corpus]
    vocab = build_word_id_dict(chain(*corpus_sents), cutoff=1)
    w2id = words_to_ids(vocab)
    return (np.array(list(map(w2id, corpus_sents))), vocab)

brown_sents = brown.sents()

brown_char_ids, brown_char_vocab = nltk_corpus_ids(brown_sents)

# HIDDEN_SIZE = 300
# 0 is reserved for padded words, so actual word indices range from 1 to
# 84, so in total there are 85 dimensions
VOCAB_SIZE = len(brown_char_vocab) + 1
OUTPUT_SIZE = VOCAB_SIZE

split = 45872  # 80% of the brown sents for training
# 4 * (len(brown_char_ids) / 5)

brown_train_ds = IndexableDataset(brown_char_ids[:split])
brown_test_ds = IndexableDataset(brown_char_ids[split:])


def char_stream(dataset, batch_size):
    # scheme = SequentialExampleScheme(dataset.num_examples)
    #     Notes on batch size: with it set to 500, my pc ran out of memory
    # with it set to 100, in the first epoch I seem to be using about 35% of
    # my RAM consistently
    batch_scheme = SequentialScheme(dataset.num_examples, batch_size=batch_size)
    just_stream = DataStream.default_stream(dataset, iteration_scheme=batch_scheme)
    #     ngrams = NGrams(1, just_stream)
    #     return ngrams
    #     return Padding(Batch(ngrams, batch_scheme))
    padded = Padding(just_stream, mask_dtype="int_")  # , mask_sources=('inputs',))
    #     return padded
    return Mapping(padded, transpose_stream)
    return just_stream


def train_model(hidden_size, batch_size):

    brown_train_stream = char_stream(brown_train_ds, batch_size)
    brown_test_stream = char_stream(brown_test_ds, batch_size)

    transition = SimpleRecurrent(name="transition",
                                 dim=hidden_size,
                                 activation=Rectifier())

    feedback = LookupFeedback(OUTPUT_SIZE,
                              feedback_dim=VOCAB_SIZE,
                              name='feedback')
    emitter = SoftmaxEmitter(name="emitter")
    readout = Readout(readout_dim=OUTPUT_SIZE,
                      source_names=["states"],
                      emitter=emitter,
                      feedback_brick=feedback,
                      name='readout')
    generator = SequenceGenerator(readout,
                                  transition,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  name='generator',)
    generator.initialize()

    inputs = tensor.lmatrix('data')
    mask = tensor.matrix('data_mask')

    cost = generator.cost(inputs, mask=mask)

    graph = ComputationGraph(cost)

    # Cost optimization
    optimizer = GradientDescent(cost=cost,
                                parameters=graph.parameters,
                                step_rule=Adam())

    # Monitoring
    monitor = DataStreamMonitoring(variables=[cost],
                                   data_stream=brown_test_stream,
                                   prefix="test")

    # Main Loop
    save_path = 'char-rnn-{}.tar'.format(hidden_size)
    main_loop = MainLoop(model=Model(cost),
                         data_stream=brown_train_stream,
                         algorithm=optimizer,
                         extensions=[monitor,
                                     FinishAfter(after_n_epochs=5),
                                     Printing(on_interrupt=True),
                                     Timing(on_interrupt=True),
                                     Checkpoint(save_path,
                                                every_n_batches=500,
                                                on_interrupt=True)
                                     # Plot("Example Plot", channels=[['test_cost_simple_xentropy', "test_error_rate"]])
                                     ])
    main_loop.run()


for h_size, b_size in ((200, 150), (300, 150), (400, 50), (500, 50)):
    logger.info("#" * 79)
    logger.info("Training with hidden size set to {} and batches of {} examples".format(h_size, b_size))
    logger.info("#" * 79)
    train_model(h_size, b_size)
