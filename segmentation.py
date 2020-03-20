"""## Spúšťaná časť"""

# @title Set up model training and evaluation  { form-width: "30%" }

# The model we explore includes three components:
# - An "Encoder" graph net, which independently encodes the edge, node, and
#   global attributes (does not compute relations etc.).
# - A "Core" graph net, which performs N rounds of processing (message-passing)
#   steps. The input to the Core is the concatenation of the Encoder's output
#   and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
#   the processing step).
# - A "Decoder" graph net, which independently decodes the edge, node, and
#   global attributes (does not compute relations etc.), on each
#   message-passing step.
#
#                     Hidden(t)   Hidden(t+1)
#                        |            ^
#           *---------*  |  *------*  |  *---------*
#           |         |  |  |      |  |  |         |
# Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
#           |         |---->|      |     |         |
#           *---------*     *------*     *---------*
#
# The model is trained by supervised learning. Input graphs are procedurally
# generated, and output graphs have the same structure with the nodes and edges
# of the shortest path labeled (using 2-element 1-hot vectors). We could have
# predicted the shortest path only by labeling either the nodes or edges, and
# that does work, but we decided to predict both to demonstrate the flexibility
# of graph nets' outputs.
#
# The training loss is computed on the output of each processing step. The
# reason for this is to encourage the model to try to solve the problem in as
# few steps as possible. It also helps make the output of intermediate steps
# more interpretable.
#
# There's no need for a separate evaluate dataset because the inputs are
# never repeated, so the training loss is the measure of performance on graphs
# from the input distribution.
#
# We also evaluate how well the models generalize to graphs which are up to
# twice as large as those on which it was trained. The loss is computed only
# on the final processing step.
#
# Variables with the suffix _tr are training parameters, and variables with the
# suffix _ge are test/generalization parameters.
#
# After around 2000-5000 training iterations the model reaches near-perfect
# performance on graphs with between 8-16 nodes.
import time

from definitionsV4 import *
from initialization import *
from accuracy import *
import os

from node2vec import Node2Vec

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data.
# Input and target placeholders.
input_ph, target_ph = create_placeholders(rand, batch_size_tr, num_nodes_min_max_tr, theta)

# Connect the data to the model.
# Instantiate the model.
# chceme vystupni velikost 2 u kazdeho uzlo, tj. tuple = 1,0 pro bilou, 0,1 pro cernou (nebo obracene, je to fuk)
model = models.EncodeProcessDecode(node_output_size=2)
# A list of outputs, one per processing step.
output_ops_tr = model(input_ph, num_processing_steps_tr)
output_ops_ge = model(input_ph, num_processing_steps_ge)

# Training loss.
loss_ops_tr = create_loss_ops(target_ph, output_ops_tr)
# Loss across processing steps.
loss_op_tr = sum(loss_ops_tr) / num_processing_steps_tr
# Test/generalization loss.
loss_ops_ge = create_loss_ops(target_ph, output_ops_ge)
loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon, use_locking)
step_op = optimizer.minimize(loss_op_tr)

# Lets an iterable of TF graphs be output from a session as NP graphs.
input_ph, target_ph = make_all_runnable_in_session(input_ph, target_ph)

# @title Reset session  { form-width: "30%" }

# This cell resets the Tensorflow session, but keeps the same computational
# graph.


last_iteration = 0
logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []
losses_ge = []
corrects_ge = []
solveds_ge = []
best_test_loss = 9999
# @title Run training  { form-width: "30%" }

# You can interrupt this cell's training loop at any time, and visualize the
# intermediate results by running the next cell (below). You can then resume
# training by simply executing this cell again.


start_time = time.time()
last_log_time = start_time
input_graphs, target_graphs = generate_networkx_graphs(rand, batch_size_tr, num_nodes_min_max_tr, theta, True)
node2vecs = []
models = []
for index in range(len(input_graphs)):
    node2vecs.append(Node2Vec(input_graphs[index], dimensions=64, walk_length=30, num_walks=200, workers=4))
    models.append(node2vecs[index].fit(window=10, min_count=1, batch_words=4))
    models[index].wv.most_similar('2')
    models[index].wv.save_word2vec_format(EMBEDDING_FILENAME = "./results/"+index++"embeddings.emb")