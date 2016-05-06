
# coding: utf-8

# Data Prep

# In[1]:

from collections import Counter
import string

from nltk.corpus import brown

from workspace import *


punctuation_symbols = set(string.punctuation)
punctuation_symbols.add("``")
punctuation_symbols.add("''")
punctuation_symbols.add('""')


def preprocess_clean(text):
    return [w.lower() for w in text if w not in punctuation_symbols]


def build_word_id_dict(word_seq, cutoff=5):
    counts = Counter(word_seq)
    frequent_enough = (word for word in counts if counts[word] >= cutoff)
    return dict((word, w_id) for w_id, word in enumerate(frequent_enough))


def convert_to_ids(word_seq, word_id_dict):
    return np.array([word_id_dict[word] for word in word_seq])


class W2VecDataset(Dataset):

    def __init__(self, sources, indexable, context_len):
        if len(sources) < 2:
            raise ValueError("Can't handle more than 2 sources atm")

        # must be len 2
        self.provides_sources = sources
        # must be a numpy array
        self.indexable = indexable
        self.N = context_len
        self.axis_labels = None

    @property
    def example_indices(self):
        return list(range(self.N, len(self.indexable) - self.N))

    @property
    def num_examples(self):
        return len(self.example_indices)

    def get_data(self, state=None, request=None):
        if state is not None or request is None:
            raise ValueError

        by_item = map(self._get_items, request)
        contexts, targets = tuple(zip(*by_item))
        return (np.array(contexts), np.array(targets))

    def _get_items(self, index):
        context_indeces = np.array(range(index - self.N, index + self.N + 1))
        # remove the index itself from context indices
        context_indeces = context_indeces[context_indeces != index]
        try:
            return (self.indexable[context_indeces], self.indexable[index])
        except IndexError:
            raise IndexError("{0}, {1}".format(str(context_indeces), str(index)))


def make_w2vec_dataset(indexable_seq):
    return W2VecDataset(('contexts', 'targets'), indexable_seq, 2)


corp_len = 10000
training_len = 4600
wrds = brown.words()[:corp_len]


# In[5]:

clean_words = preprocess_clean(wrds)

vocab = build_word_id_dict(clean_words)

filtered_words = [w for w in clean_words if w in vocab]

word_ids = convert_to_ids(filtered_words, vocab)


# In[6]:

training_dataset = make_w2vec_dataset(word_ids[:training_len])
test_dataset = make_w2vec_dataset(word_ids[training_len:])


# In[7]:

len(test_dataset.indexable)


# In[8]:

test_dataset.example_indices[-1]


# In[9]:

training_dataset.indexable[4597:]


# In[10]:

len(training_dataset.indexable)


# In[11]:

str(np.array([1, 2, 3]))


# In[13]:

training_dataset.get_data(request=[0])


# Network Definition
#
# Revisit regularization: how does it work?
#

# In[31]:


# In[16]:

sequence_generator2.SequenceGenerator


# In[33]:

hidden_size = 500
vocab_size = len(vocab)

# Network layers
# Not sure I should keep this as input?
input_layer = tensor.imatrix('contexts')
input_to_projection = LookupTable(vocab_size,
                                  hidden_size,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  name="projection")
projection_layer = tensor.mean(input_to_projection.apply(input_layer), axis=1)
projection_layer.name = 'projection'
projection_to_ouput = Linear(name='output',
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0),
                             input_dim=hidden_size,
                             output_dim=vocab_size)
probs = Softmax().apply(projection_to_ouput.apply(projection_layer))

# Cost Function, Graph
true_targets = tensor.ivector('targets')
cost = CategoricalCrossEntropy(name='simple_entropy').apply(true_targets, probs)
graph = ComputationGraph(cost)

# Other metrics
# not sure this will work...
# error_rate = MisclassificationRate().apply(probs, true_targets)

# Parameter Initialization
# Idea: annotate layers that need initialization and select them
input_to_projection.initialize()
projection_to_ouput.initialize()

# Cost optimization
optimizer = GradientDescent(cost=cost, parameters=graph.parameters,
                            #                             step_rule=Scale(learning_rate=0.025),
                            step_rule=Adam()
                            )

# Data Streams
training_stream = DataStream.default_stream(training_dataset,
                                            iteration_scheme=SequentialScheme(training_dataset.example_indices, batch_size=200))
test_stream = DataStream.default_stream(test_dataset,
                                        iteration_scheme=SequentialScheme(test_dataset.example_indices, batch_size=20))
# Monitoring
monitor = DataStreamMonitoring(variables=[cost],
                               data_stream=test_stream, prefix="test")

# Main Loop
main_loop = MainLoop(data_stream=training_stream, algorithm=optimizer,
                     extensions=[monitor,
                                 FinishAfter(after_n_epochs=1),
                                 Printing(),
                                 Plot("Example Plot",
                                      channels=[
                                          ['test_simple_entropy_apply_cost', "test_error_rate"]],
                                      after_batch=True)
                                 ])


if __name__ == '__main__':
    main_loop.run()
