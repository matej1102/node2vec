"""Tu sa nachádzajú definície ku spúštaciemu súboru segmentation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from initialization import *
from scipy.spatial.distance import dice

import collections
import itertools
import time
import time

import tensorflow as tf
import graph_nets

from pygments.lexer import include, combined

import networkx as nx
import numpy as np

tf.logging = tf.compat.v1.logging

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models

import glob

import imageio

"""## Definície ku grafom"""

import collections
import itertools
import time

from pygments.lexer import include, combined

from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import networkx as nx
import numpy as np

import glob

import imageio

SEED = 10
np.random.seed(SEED)
tf.set_random_seed(SEED)

# @title Helper functions  { form-width: "30%" }

# pylint: disable=redefined-outer-name

DISTANCE_WEIGHT_NAME = "distance"  # The name for the distance edge attribute.


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))


def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def get_node_dict(graph, attr):
    """Return a `dict` of node:attribute pairs from a graph."""
    return {k: v[attr] for k, v in graph.node.items()}


def slice_blocks(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))
def add_arrays(array1,array2):
    return np.concatenate([array1,array2],0)

def add_edges_to_graph(act_node_num, graph, img, row, column,region,image_solution):
    act_node = img[row][column]
    # sousedni hrany
    neighbours = []
    if row == region-1:
        if column <= region-1:
            neighbours = ['right_node', 'lower_node']
            # neighbours = ['right_node']
        elif column >= 128-region:
            neighbours = ['left_node', 'lower_node']
            # neighbours = ['left_node']
        else:
            neighbours = ['right_node', 'left_node', 'lower_node']
            # neighbours = ['right_node']
    elif row >= 128-region:
        if column <= region-1:
            neighbours = ['right_node', 'upper_node']
        #  neighbours = ['right_node']
        elif column >= 128-region:
            neighbours = ['left_node', 'upper_node']
            # neighbours = ['left_node']
        else:
            neighbours = ['right_node', 'left_node', 'upper_node']
    #     neighbours = ['right_node']
    elif column <= region-1:
        neighbours = ['right_node', 'upper_node', 'lower_node', ]
    #     neighbours = ['right_node',]
    elif column >= 128-region:
        neighbours = ['left_node', 'upper_node', 'lower_node']
    #     neighbours = ['left_node']
    else:
        neighbours = ['right_node', 'left_node', 'lower_node', 'upper_node']
    #     neighbours = ['right_node']
    neighbours, nodes_number = generate_nodes(neighbours, img, row, column,region,image_solution)

    for node, node_num in zip(neighbours, nodes_number):
        a = act_node
        b = node

        weight = pow((255 - abs(a - b)) / 255, 2)
        #toto nefunguje pretože je váha nastavená 0-255 pow((255 - abs(a - b)) / 255, 2)
        graph.add_edge(act_node_num, node_num, weight=weight, solution=0)

        # graph.add_edge(node_num, act_node_num, weight=weight, solution=0)


def generate_nodes(neighbours, img, row, column,region,img_solution):
    nodes = []
    nodes_number = []
    edge_solutions= []
    row_pixels = 128
    for neighbour in neighbours:
        if neighbour == 'right_node':
            nodes.append(img[row][column + region])
            nodes_number.append(row * row_pixels + column + 1)
            #edge_solutions.append(img_solution[row][column]==img_solution[row][column+region])
            continue
        elif neighbour == 'left_node':
            nodes.append(img[row][column - region])
            nodes_number.append(row * row_pixels + column - 1)
            #edge_solutions.append(img_solution[row][column] == img_solution[row][column - region])
            continue
        elif neighbour == 'upper_node':
            nodes.append(img[row - region][column])
            nodes_number.append((row - 1) * row_pixels + column)
            #edge_solutions.append(img_solution[row][column] == img_solution[row-region][column])
            continue
        elif neighbour == 'lower_node':
            nodes.append(img[row + region][column])
            nodes_number.append((row + 1) * row_pixels + column)
            #edge_solutions.append(img_solution[row][column] == img_solution[row + region][column])
            continue
        elif neighbour == 'upper_right_node':
            nodes.append(img[row - region][column + region])
            nodes_number.append((row - 1) * row_pixels + column + 1)
            #edge_solutions.append(img_solution[row][column] == img_solution[row - region][column + region])
            continue
        elif neighbour == 'upper_left_node':
            nodes.append(img[row - region][column - region])
            nodes_number.append((row - 1) * row_pixels + column - 1)
            #edge_solutions.append(img_solution[row][column] == img_solution[row - region][column - region])
            continue
        elif neighbour == 'lower_left_node':
            nodes.append(img[row + region][column - region])
            nodes_number.append((row + 1) * row_pixels + column - 1)
            #edge_solutions.append(img_solution[row][column] == img_solution[row + region][column - region])
            continue
        else:
            nodes.append(img[row + region][column + region])
            nodes_number.append((row + 1) * row_pixels + column + 1)
            #edge_solutions.append(img_solution[row][column] == img_solution[row + region][column + region])
            continue

    return nodes, nodes_number #,edge_solutions


def generate_graphs(img_path, solution_path,region, j, placeholder=False):
    img = imageio.imread(img_path)
    img_solution = imageio.imread(solution_path)

    img = np.array(img)
    img_solution = np.array(img_solution)
    img_subsample = img[::2, ::2]
    img_solution_subsample = img_solution[::2, ::2]

    images = slice_blocks(img_subsample, 128, 128)
    #images = add_arrays(images, slice_blocks(img, 128, 128))

    images_solutions = slice_blocks(img_solution_subsample, 128, 128)
    images_solutions = add_arrays(images_solutions, slice_blocks(img_solution, 128, 128))
    # if placeholder == True:
    #    index = j * 4096 - 1
    #   j2 = j + 1

    # else:
    #
    # j2 = 1

    graphs = [];
    # iterace radky a sloupce
    for i in range(len(images)):
        index = 0 * 4096 - 1
        graph = nx.Graph()
        file = open("./results/"+os.path.basename(os.path.dirname(solution_path))+"_"+str(i)+"solution.txt", "a")
        for row in range(0, 128):
            for column in range(0, 128):
                index += 1

                node_solution = 0
                value = images[i][row][column] / 255
                if images_solutions[i][row][column] == 255:
                    node_solution = 1
                    # pridani uzlu
                file.write(str(index)+"\t"+str(node_solution)+"\n")
                graph.add_node(index, value=value, solution=node_solution)
                # pridani hran
                add_edges_to_graph(index, graph, img, row, column,region,images_solutions[i])
        graphs.append(graph)
        file.close()
    return graphs


def graph_to_input_target(graph):
    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("value",)
    input_edge_fields = ("weight",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        solution_length = 0
        # print(node_index)
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields))

        target_graph.add_node(node_index, features=create_feature(node_feature, target_node_fields))
        target_node = to_one_hot(
            create_feature(node_feature, target_node_fields).astype(int), 2)[0]
        target_graph.add_node(node_index, features=target_node)
        solution_length += 1
    solution_length = graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = to_one_hot(
            create_feature(features, target_edge_fields).astype(int), 2)[0]
        target_graph.add_edge(sender, receiver, features=target_edge)
    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph


def generate_networkx_graphs(rand, num_examples, num_nodes_min_max, theta, placeholder=False , training=True):
    """Generate graphs for training.

    Args:
      rand: A random seed (np.RandomState instance).
      num_examples: Total number of graphs to generate.
      num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
        graph. The number of nodes for a graph is uniformly sampled within this
        range.
      theta: (optional) A `float` threshold parameters for the geographic
        threshold graph's threshold. Default= the number of nodes.

    Returns:
      input_graphs: The list of input graphs.
      target_graphs: The list of output graphs.
      graphs: The list of generated graphs.
    """
    if training:
        low = train_start_image
        high = train_end_image
        # high=24
    else:
        low = test_start_image
        high = test_end_image
    input_images = []
    target_images = []
    input_graphs = []
    target_graphs = []

    for image_path, solution_path in zip(glob.glob("dataset/images/dir/*.png"), glob.glob("dataset/labels/dir/*.png")):
        input_images.append(image_path)
        target_images.append(solution_path)
    for i in range(low, high):
        # vytvoreni grafu
        # for row in range (0,8):
        j = 1
        print(i)
        graphs = generate_graphs(input_images[i], target_images[i], j, placeholder)
        for index in range(len(graphs)):
            input_graph, target_graph = graph_to_input_target(graphs[index])
            input_graphs.append(input_graph)
            target_graphs.append(target_graph)
    return input_graphs, target_graphs



def create_loss_ops(target_op, output_ops):
    """Create supervised loss operations from targets and outputs.

    Args:
      target_op: The target velocity tf.Tensor.
      output_ops: The list of output graphs from the model.

    Returns:
      A list of loss values (tf.Tensor), one per output op.
    """

    #loss_ops = [
    #     tf.losses.mean_squared_error(target_op.nodes, output_op.nodes)
    #     for output_op in output_ops
    # ]

    loss_ops = [
        tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes)
        for output_op in output_ops
    ]

    return loss_ops

def dice_loss(labels,predictions):
  numerator = 2 * tf.reduce_sum(labels * predictions, axis=-1)
  denominator = tf.reduce_sum(labels + predictions, axis=-1)

  result= 1 - (numerator + 1) / (denominator + 1)

  return result

def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]