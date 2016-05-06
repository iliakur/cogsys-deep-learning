
# coding: utf-8

# ## Data Preparation

# In[1]:

get_ipython().magic('doctest_mode')


# In[2]:

import pandas

raw_data = pandas.read_csv("agaricus-lepiota.data", header=None)
split = 2031

data_one_hot = pandas.get_dummies(raw_data)

training_one_hot = (data_one_hot.iloc[split:, 2:], data_one_hot.iloc[split:, :2])

testing_one_hot = (data_one_hot.iloc[:split, 2:], data_one_hot.iloc[:split, :2])

from fuel.datasets import IndexableDataset
training_dataset = IndexableDataset(
    indexables={'features': training_one_hot[0].values.astype('i8'), 'targets': training_one_hot[1].values.astype('i8')})
testing_dataset = IndexableDataset(
    indexables={'features': testing_one_hot[0].values.astype('i8'), 'targets': testing_one_hot[1].values.astype('i8')})


# ## Blocks Tutorial

# In[3]:

import theano
# theano.config.optimizer = "None"
# theano.config.exception_verbosity = "high"


# In[4]:

from theano import tensor
x = tensor.lmatrix('features')


# In[5]:

from blocks.bricks import Linear, Logistic, Softmax


# In[10]:

hidden_layer_size = 100
input_to_hidden = Linear(name='input_to_hidden', input_dim=117, output_dim=hidden_layer_size)
h = Logistic().apply(input_to_hidden.apply(x))
hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_layer_size, output_dim=2)
y_hat = Softmax().apply(hidden_to_output.apply(h))

y = tensor.lmatrix('targets')
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
cost = CategoricalCrossEntropy().apply(y, y_hat)
error_rate = MisclassificationRate().apply(y.argmax(axis=1), y_hat)
error_rate.name = "error_rate"

# >>> from blocks.roles import WEIGHT
from blocks.graph import ComputationGraph
# >>> from blocks.filter import VariableFilter
cg = ComputationGraph(cost)
# >>> W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
# >>> cost = cost + 0.005 * (W1 ** 2).sum() + 0.005 * (W2 ** 2).sum()
# >>> cost.name = 'cost_with_regularization'
cost.name = 'cost_simple_xentropy'

from blocks.initialization import IsotropicGaussian, Constant
input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_output.initialize()

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, SequentialExampleScheme
# >>> from fuel.transformers import Flatten
data_stream = DataStream.default_stream(
    training_dataset,
    iteration_scheme=SequentialScheme(training_dataset.num_examples, batch_size=20))

data_stream_test = DataStream.default_stream(
    testing_dataset,
    iteration_scheme=SequentialScheme(testing_dataset.num_examples, batch_size=split))

from blocks.extensions.monitoring import DataStreamMonitoring
monitor = DataStreamMonitoring(
    variables=[cost, error_rate], data_stream=data_stream_test, prefix="test")


# In[11]:

get_ipython().magic('pinfo DataStreamMonitoring')


# In[ ]:




# In[8]:

cost.type


# In[7]:

from blocks.algorithms import GradientDescent, Scale
algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.025))

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks_extras.extensions.plot import Plot
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[monitor,
                                     FinishAfter(after_n_epochs=3),
                                     Printing(),
                                     Plot("Example Plot", channels=[['test_cost_simple_xentropy', "test_error_rate"]])
                                ])
main_loop.run()


# ### Observations:
# 
# **VERY IMPORTANT**
# there's some sort of shared state going on in the model definition, so it's important to rerun all the code from the beginning, not just the main loop!
# 
# Setting the hidden layer to 50 lowered the cost (0.69), but didn't improve the score after more training.
# Setting the hidden layer to 300 bumped up the cost (1.79), but training made significant improvements in it after first epoch, but not in subsequent ones (more incremental after that).
# 
# Somehow the total number of epochs influences the starting cost??
# 
# - epochs: 5 vs 3
# - learning rate: 0.5
# - hidden layer: 300
# 
# When I took the same parameters (5 epochs) and set hidden layer to 100, I got the following progression of costs:
# 
# - epochs done: 0 = 0.6931921183574188
# - epochs done: 1 = 1.814269964941275
# - epochs done: 2 = 1.194301165186615
# - epochs done: 3 = 0.8182568883881371
# - epochs done: 4 = 0.7323559855023634
# - epochs done: 5 = 0.6993469372860405
# 
# 
# ### Data Processing
# 
# What is `Flatten` for?
# 
# ### Minibatches and Train/Test Split
# 
# What's the relationship between the test/training data and the minibatch size?
# Does the batch size have to "fit" exactly into the dataset sizes?
# Why are we also iterating over the test data?
# 
# 

# ## From Theano intro tutorial

# In[1]:

import theano
from theano import tensoror


# In[7]:

a = tensor.dscalar("a")
b = tensor.dscalar("b")


# In[8]:

c = a + b
f = theano.function([a, b], c)


# In[4]:

assert 4 == f(1.5, 2.5)


# In[9]:

theano.pp(c)


# In[16]:

c.owner.op.name

