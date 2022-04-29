#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DAG related functionalities.

This file contains the tools to treat the DAG used in the DATGAN.
"""
import copy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder


def advise(data, dag, plot_graphs=False):
    """
    Give advice about which edge could be added in the DAG based on Pearson and Spearman correlations

    Parameters
    ----------
    data: pandas.DataFrame
        Original dataset
    dag: networkx.DiGraph
        Directed Acyclic Graph representing the relations between the variables
    plot_graphs: bool
        Whether to plot some graphs or not
    """

    print("Preparing advice...")

    # First, we transform the string values in the DF into numerical values
    num_df = {}

    cat_encoder = LabelEncoder()

    for c in data.columns:
        if data[c].dtype == 'object':
            num_df[c] = cat_encoder.fit_transform(data[c])
        else:
            num_df[c] = data[c]

    num_df = pd.DataFrame(num_df)

    # Compute the Pearson and Spearman correlations for each combination of columns
    cols = np.array(data.columns)
    corr_p = np.zeros((len(cols), len(cols)))
    corr_s = np.zeros((len(cols), len(cols)))

    for x in combinations(enumerate(cols), 2):
        i, ci = x[0]
        j, cj = x[1]
        val, _ = pearsonr(num_df[ci], num_df[cj])
        # We take the absolute value since we only care about the magnitude
        corr_p[i, j] = np.abs(val)
        corr_p[j, i] = corr_p[i, j]
        val, _ = spearmanr(num_df[ci], num_df[cj])
        corr_s[i, j] = np.abs(val)
        corr_s[j, i] = corr_s[i, j]

    # Now, we check the correlation values for each of the edges that have already been added in the DAG
    corr_p_vals = []
    corr_s_vals = []

    for e in dag.edges:
        ci = e[0]
        cj = e[1]

        i = np.where(cols == ci)[0][0]
        j = np.where(cols == cj)[0][0]

        corr_p_vals.append(corr_p[i, j])

        # Put the values to 0 to not count them later on
        corr_p[i, j] = 0
        corr_p[j, i] = 0

        corr_s_vals.append(corr_s[i, j])

        # Put the values to 0 to not count them later on
        corr_s[i, j] = 0
        corr_s[j, i] = 0

    # Set all the correlation values to 0 in the lower triangular part of the matrices (avoid double values)
    for i in range(len(cols)):
        for j in range(i):
            corr_p[i, j] = 0
            corr_s[i, j] = 0

    # Get the name of the edges in strings
    links_p = get_link_names(corr_p_vals, corr_p, cols)
    links_s = get_link_names(corr_s_vals, corr_s, cols)

    # Check which edges have both a high pearson and a high Spearman correlation
    final_links = list(set(links_s).intersection(set(links_p)))

    print("You might want to add the following edges in your DAG (direction not given here):")
    for link in final_links:
        print("  - {} <-> {}".format(link[0], link[1]))

    if plot_graphs:

        for link in final_links:
            a = num_df[link[0]]
            b = num_df[link[1]]

            counts = Counter([(x, y) for x, y in zip(a, b)])
            points = set([(x, y) for x, y in zip(a, b)])
            a = list()
            b = list()
            for x, y in points:
                a.append(x)
                b.append(y)

            size = [counts[(x, y)] for x, y in zip(a, b)]

            plt.figure(figsize=(10, 7))
            plt.scatter(a, b, size, color='k')
            plt.xlabel(link[0])
            plt.ylabel(link[1])
            if len(np.unique(a)) < 10:
                plt.xticks(np.unique(a))
            if len(np.unique(b)) < 10:
                plt.yticks(np.unique(b))


def get_link_names(values, matrix, names, n=10):
    """
    Compute a list of links with the string names

    Parameters
    ----------
    values: list
        Existing correlation values
    matrix: np.ndarray
        All the remaining correlation values
    names: np.ndarray
        List of string names
    n: int
        Maximum number of links returned

    Returns
    -------
    links: list[str]
        List of edges with the string name

    """
    thresh = np.quantile(values, 0.75)

    values = np.sort(np.partition(np.asarray(matrix), matrix.size - n, axis=None)[-n:])
    values = values[values >= thresh]

    tmp = [np.where(matrix == v) for v in values]

    links = []

    for x in tmp:
        i = x[0][0]
        j = x[1][0]

        links.append((names[i], names[j]))

    return links


def verify_dag(data, dag):
    """
    Verify the integrity of the DAG

    Parameters
    ----------
    data: pandas.DataFrame
        Original dataset
    dag: networkx.DiGraph
        Directed Acyclic Graph representing the relations between the variables

    Raises
    ------
    TypeError:
        If the variable `dag` is not of the class networkx.DiGraph.
    ValueError:
        If the DAG has cycles
    ValueError:
        If the number of nodes in the DAG does not correspond to the number of variables in the dataset.
    """

    # 1. Verify the type
    if type(dag) is not nx.classes.digraph.DiGraph:
        raise TypeError("Provided graph is not from the type \"networkx.classes.digraph."
                        "DiGraph\": {}".format(type(dag)))

    # 2. Verify that the graph is indeed a DAG
    if not nx.algorithms.dag.is_directed_acyclic_graph(dag):

        cycles = nx.algorithms.cycles.find_cycle(dag)

        if len(cycles) > 0:
            raise ValueError("Provided graph is not a DAG. Cycles found: {}".format(cycles))
        else:
            raise ValueError("Provided graph is not a DAG.")

    # 3. Verify that the dag has the correct number of nodes
    if len(dag.nodes) != len(data.columns):
        raise ValueError("DAG does not have the same number of nodes ({}) as the number of "
                         "variables in the data ({}).".format(len(dag.nodes), len(data.columns)))


def get_in_edges(dag):
    """
    Return the in-edges for each node in the DAG.

    Parameters
    ----------
    dag: networkx.DiGraph
        Directed Acyclic Graph representing the relations between the variables

    Returns
    -------
    in_edges: dct
        Dictionary of in-edges for each node in the DAG
    """
    # Get the in_edges
    in_edges = {}

    for n in dag.nodes:
        in_edges[n] = []
        for edge in dag.in_edges:
            if edge[1] == n:
                in_edges[n].append(edge[0])

    return in_edges


def get_order_variables(dag):
    """
    Compute the order of the variables used when creating the structure of the Generator.

    Parameters
    ----------
    dag: networkx.DiGraph
        Directed Acyclic Graph representing the relations between the variables

    Returns
    -------
    treated: list[str]
        Ordered list of the variable names
    n_sources: int
        Number of sources in the DAG
    """

    # Get the in_edges
    in_edges = get_in_edges(dag)

    untreated = set(dag.nodes)
    treated = []

    # Get all nodes with 0 in degree
    to_treat = [node for node, in_degree in dag.in_degree() if in_degree == 0]

    n_sources = len(to_treat)

    while len(untreated) > 0:
        # remove the treated nodes
        for n in to_treat:
            untreated.remove(n)
            treated.append(n)

        to_treat = []
        # Find the edges that are coming from the the treated nodes
        for edge in dag.in_edges:

            all_treated = True
            for l in in_edges[edge[1]]:
                if l not in treated:
                    all_treated = False

            if edge[0] in treated and all_treated and edge[1] not in treated and edge[1] not in to_treat:
                to_treat.append(edge[1])

    return treated, n_sources


def linear_dag(data, conditional_inputs):
    """
    Return a linear graph for the DATGAN. Each column is connected to the next one

    Parameters
    ----------
    data: pandas.DataFrame
        original data
    conditional_inputs: list[str]
        List of variable names used as conditional inputs

    Returns
    -------
    graph: networkx.DiGraph
        Linear DAG
    """

    graph = nx.DiGraph()

    non_cond_vars = list(set(data.columns) - set(conditional_inputs))

    list_ = []

    for var in conditional_inputs:
        list_.append((var, non_cond_vars[0]))

    for i in range(len(non_cond_vars) - 1):
        list_.append((non_cond_vars[i], non_cond_vars[i + 1]))

    graph.add_edges_from(list_)

    return graph


def transform_dag(dag, cond_inputs):
    """
    If we have some conditional inputs, we want to treat these values as nodes => we need to reverse some of the links
    in the DAG.

    Parameters
    ----------
    dag: networkx.DiGraph
        Original DAG provided by the user
    cond_inputs: list[str]
        List of node names that are treated as conditional inputs

    Returns
    -------
    dag: networkx.DiGraph
        Updated version of the DAG

    """
    # Reverse nodes
    for node in cond_inputs:
        list_ = copy.deepcopy(dag.in_edges(node))

        for e in list_:
            if (e[0] not in cond_inputs):
                dag.add_edges_from([e[::-1]])
                dag.remove_edges_from([e])

        del list_

    return dag