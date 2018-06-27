##
#	Biaxial Recurrent Neural Network for Music Composition
#
## This code implements a recurrent neural network trained to generate classical music. The model, which uses LSTM layers and draws inspiration from convolutional neural networks, learns to predict which notes will
## be played at each time step of a musical piece.
#
# You can read about its design and hear examples on this blog post.
#	 Requirements
#
#
##This code is written in Python, and depends on having Theano and theano-lstm (which can be installed with pip) installed. The bare minimum you should need to do to get everything running, assuming you have Python, is
#
#	sudo pip install --upgrade theano
#	sudo pip install numpy scipy theano-lstm python-midi
#
#
#
# create model class instance
# where the numbers are the sizes of the hidden layers in
# the two parts of the network architecture. This will
# take a while, as this is where Theano will compile its optimized functions.
import model
m = model.Model([300,300],[100,50], dropout=0.5) 

# load data out of the music folder
import multi_training
pcs = multi_training.loadPieces("music")

# start training
# This will train using 10000 batches of 10 eight-measure
# segments at a time, and output a sampled output and the learned parameters
# every 500 iterations.
multi_training.trainPiece(m, pcs, 10000)

# you can generate a full composition after training is complete.
# The function gen_adaptive in main.py will generate a piece and also prevent
# long empty gaps by increasing note probabilities if the network stops playing
# for too long.

gen_adaptive(m,pcs,10,name="komposition")

# There are also mechanisms to observe the hidden activations and memory cells
# of the network, but these are still a work in progress at the moment.
#
# Right now, there is no separate validation step, because my initial goal was
# to produce interesting music, not to assess the accuracy of this method.
# It does, however, print out the cost on the training set after every
# 100 iterations during training.
#
# If you want to save your model weights, you can do
#	pickle.dump( m.learned_config, open( "path_to_weight_file.p", "wb" ) )
#
# And if you want to load them, do
#	m.learned_config = pickle.load(open( "path_to_weight_file.p", "rb" ) )
#

