#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DAG related functionalities.

This file contains the tools to treat the DAG used in the DATGAN.
"""

import networkx as nx


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
    """

    # Get the in_edges
    in_edges = get_in_edges(dag)

    untreated = set(dag.nodes)
    treated = []

    # Get all nodes with 0 in degree
    to_treat = [node for node, in_degree in dag.in_degree() if in_degree == 0]

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

    return treated
