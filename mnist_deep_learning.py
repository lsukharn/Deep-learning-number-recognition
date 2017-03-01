#!/usr/bin/env python
from neon.util.argparser import NeonArgparser

parser = NeonArgparser(__doc__)
args = parser.parse_args()

from neon.data import MNIST
from neon.data import ArrayIterator

mnist = MNIST()

(X_train, y_train), (X_test, y_test), nclass = mnist.load_data()

#print "X_test: %s" % X_test[1]

# setup training set iterator
train_set = ArrayIterator(X_train, y_train, nclass=nclass)
# setup test set iterator
test_set = ArrayIterator(X_test, y_test, nclass=nclass)

#Initialize weights to small random numbers with Gaussian
from neon.initializers import Gaussian

init_norm = Gaussian(loc=0.0, scale=0.01)

#Affine is a FC network with 100 hidden units
from neon.layers import Affine
#We will use ReLu for hidden units and SoftMax fro output units. Softmax is used to ensure that 
#all outputs sum up to 1 and are within the range of [0, 1]
from neon.transforms import Rectlin, Softmax

layers = []
layers.append(Affine(nout=100, init=init_norm, activation=Rectlin()))
layers.append(Affine(nout=10, init=init_norm,
                     activation=Softmax()))

# initialize model object
from neon.models import Model

mlp = Model(layers=layers)

#The cost function is wrapped within a GeneralizedCost layer, which handles the comparison of the #outputs with the provided labels in the dataset.
#Get cost function, CrossEntropyMulti
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
#For learning, we use stochastic gradient descent with a learning rate of 0.1 and momentum #coefficient of 0.9.
from neon.optimizers import GradientDescentMomentum

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)
#callbacks show the progress of calculations
from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

#train the model
mlp.fit(train_set, optimizer=optimizer, num_epochs=10, cost=cost,
        callbacks=callbacks)
#The variable results is a numpy array with 
#shape (num_test_examples, num_outputs) = (10000,10) with the model probabilities for each label.
results = mlp.get_outputs(test_set)

print "labels: %s, %s, %s: " % (y_test[2], y_test[5], y_test[100])
#ind1, val1 = max(results[2].tolist())
#ind2, val2 = max(results[5].tolist())
#ind3, val3 = max(results[100].tolist())
print "results: %s, %s, %s:" % (results[2].tolist(), results[5].tolist(), results[100].tolist())

from neon.transforms import Misclassification

# evaluate the model on test_set using the misclassification metric
error = mlp.eval(test_set, metric=Misclassification())*100
print('Misclassification error = %.1f%%' % error)

