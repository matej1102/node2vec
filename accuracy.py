import  numpy as np
from scipy.spatial.distance import *

from graph_nets import utils_np
from initialization import *


def compute_jaccard_accuracyTest(target, output, iteration, index, test_values, last_iteration,iteration_count, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
      target: A `graphs.GraphsTuple` that contains the target graph.
      output: A `graphs.GraphsTuple` that contains the output graph.
      use_nodes: A `bool` indicator of whether to compute node accuracy or not.
      use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
      correct: A `float` fraction of correctly labeled nodes/edges.
      solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
      ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        solved_pixels = []

        if iteration_count<int(iteration/log_every_outputs):
            for i in range(len(test_values)):
                with open('results/'+str(last_iteration) + '_' + str(index) + '.txt', 'w+') as f:
                    for value in yn:
                        f.write("%s\n" % value)
                    f.close()
                with open('results/'+'target_' + str(last_iteration) + '_' + str(index) + '.txt', 'w+') as x:
                    for value in xn:
                        x.write("%s\n" % value)


        if use_nodes:
            c.append(float(1 - dice(xn, yn)))
            solved_pixels.append(xn == yn)
        if use_edges:
            c.append(float(1 - dice(xe, ye)))
            solved_pixels.append(xe == ye)
        s = np.all(solved_pixels)
        cs.append(c)
        ss.append(s)
    mean_koeficient = np.mean(cs)
    solved = np.mean(np.stack(ss))
    return mean_koeficient, solved

def compute_jaccard_accuracy(target, output, use_nodes=True,use_edges=False):
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []

    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        solved_pixels=[]
        if use_nodes:
            c.append(float(1-jaccard(xn,yn)))
            solved_pixels.append(xn == yn)
        if use_edges:
            c.append(float(1-jaccard(xe,ye)))
            solved_pixels.append(xe == ye)
        "c = np.concatenate(c, axis=0)"
        s = np.all(solved_pixels)
        cs.append(c)
        ss.append(s)
    mean_koeficient = np.mean(cs)
    solved = np.mean(np.stack(ss))
    return mean_koeficient, solved
def compute_dice_accuracyTest(target, output, iteration, index, test_values, last_iteration,iteration_count, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
      target: A `graphs.GraphsTuple` that contains the target graph.
      output: A `graphs.GraphsTuple` that contains the output graph.
      use_nodes: A `bool` indicator of whether to compute node accuracy or not.
      use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
      correct: A `float` fraction of correctly labeled nodes/edges.
      solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
      ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        solved_pixels = []

        if iteration_count<int(iteration/log_every_outputs):
            for i in range(len(test_values)):
                with open('results/'+str(last_iteration) + '_' + str(index) + '.txt', 'w+') as f:
                    for value in yn:
                        f.write("%s\n" % value)
                    f.close()
                with open('results/'+'target_' + str(last_iteration) + '_' + str(index) + '.txt', 'w+') as x:
                    for value in xn:
                        x.write("%s\n" % value)


        if use_nodes:
            c.append(float(1 - dice(xn, yn)))
            solved_pixels.append(xn == yn)
        if use_edges:
            c.append(float(1 - dice(xe, ye)))
            solved_pixels.append(xe == ye)
        s = np.all(solved_pixels)
        cs.append(c)
        ss.append(s)
    mean_koeficient = np.mean(cs)
    solved = np.mean(np.stack(ss))
    return mean_koeficient, solved

def compute_dice_accuracy(target, output, use_nodes=True,use_edges=False):
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []

    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        solved_pixels=[]
        if use_nodes:
            c.append(float(1-dice(xn,yn)))
            solved_pixels.append(xn == yn)
        if use_edges:
            c.append(float(1-dice(xe,ye)))
            solved_pixels.append(xe == ye)
        "c = np.concatenate(c, axis=0)"
        s = np.all(solved_pixels)
        cs.append(c)
        ss.append(s)
    mean_koeficient = np.mean(cs)
    solved = np.mean(np.stack(ss))
    return mean_koeficient, solved

def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
      target: A `graphs.GraphsTuple` that contains the target graph.
      output: A `graphs.GraphsTuple` that contains the output graph.
      use_nodes: A `bool` indicator of whether to compute node accuracy or not.
      use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
      correct: A `float` fraction of correctly labeled nodes/edges.
      solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
      ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []

    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved

def compute_accuracyTest(target, output, iteration, index, test_values, last_iteration, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
      target: A `graphs.GraphsTuple` that contains the target graph.
      output: A `graphs.GraphsTuple` that contains the output graph.
      use_nodes: A `bool` indicator of whether to compute node accuracy or not.
      use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
      correct: A `float` fraction of correctly labeled nodes/edges.
      solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
      ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if iteration % log_every_outputs == 0:
            for i in range(len(test_values)):
                with open(str(last_iteration) + '_' + str(index) + '.txt', 'w+') as f:
                    for value in yn:
                        f.write("%s\n" % value)
                    f.close()
                with open('target_' + str(last_iteration) + '_' + str(index) + '.txt', 'w+') as x:
                    for value in xn:
                        x.write("%s\n" % value)

        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=-1)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=-1))
    solved = np.mean(np.stack(ss))
    return correct, solved
