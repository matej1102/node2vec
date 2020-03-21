"""Inicializačné parametre k segmentácii"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

seed = 10
rand = np.random.RandomState(seed=seed)

# Model parameters.
# Number of processing (message-passing) steps.
num_processing_steps_tr = 6
num_processing_steps_ge = 6

# Data / training parameters.
num_training_iterations = 100
theta = 3
# Large values (1000+) make trees. Try 20-60 for good non-trees.
batch_size_tr = 1
batch_size_ge = 1
# Number of nodes per graph sampled uniformly from this range.
num_nodes_min_max_tr = (8, 17)
num_nodes_min_max_ge = (16, 33)

# Optimizer.
learning_rate = 5e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 2e-10  # pôvodne 1e-08
use_locking = False
# Starting and ending image used for training
train_start_image=0 #minimum 0
train_end_image=1 #maximum 24
# Starting and ending image used for testing
test_start_image=24 #minimum 24
test_end_image=27  #mmaximum 29
# Time of logging into console
log_every_seconds = 180
# Log and output on X iterations
iteration_count=0
log_every_outputs = 30
#region of edges
region=1