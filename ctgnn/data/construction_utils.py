"""
This module contains helper functions for the construction of graphs.
"""

from typing import Any, Callable

import numpy as np
from torch_geometric.data import Data

from ctgnn.data.graph_construction import construct_graph

__all__ = ["get_constructed_graph"]


def get_constructed_graph(
    constr_method: str,
    desired_avg_degree: int = 7,
    verbose: bool = True,
    **same_params: Any,
) -> tuple[Data, dict]:
    """Construct a graph from the dataset of this searcher
    that is close to the `desired_avg_degree`.
    If the degree cannot be specified for a construction method,
    the parameter will be optimized using `construct_graph_by_degree`.

    Args:
        constr_method: The construction method as string
        desired_avg_degree: The desired average degree of the graph. Defaults to 7.

    Returns:
        A tuple of the graph and the parameter value
        that was optimized towards the desired average degree as dictionary
    """

    if "self_loops" not in same_params:
        same_params["self_loops"] = True

    if constr_method == "knn":
        # TODO: CHANGE TO ONLY SUBTRACT 1 REGARDLESS OF SELF-LOOPS SINCE THE IMPLEMENTATION CHANGED
        # In total -2: -1 for self-loops and -1 for the conversion from directed
        # to undirected since this adds some additional edges
        k = (
            desired_avg_degree - 2
            if same_params["self_loops"]
            else desired_avg_degree - 1
        )
        graph = construct_graph(algorithm="knn", param=k, **same_params)
        degree, _ = calc_deg(x=0, graph_func=lambda var: graph)
        return graph, {"k": k, "degree": degree}
    if constr_method == "delaunay":
        graph = construct_graph(algorithm="delaunay", **same_params)
        degree, _ = calc_deg(x=0, graph_func=lambda var: graph)
        return graph, {"degree": degree}
    if constr_method == "radius":
        graph, radius, degree = construct_graph_by_degree(
            graph_func=lambda x: construct_graph(
                algorithm="radius", param=x, **same_params
            ),
            desired_avg_degree=desired_avg_degree,
            tolerance=0.2,
            x0=1 / len(same_params["positions"]),
            verbose=verbose,
        )
        return graph, {"radius": radius, "degree": degree}
    if constr_method == "empty":
        graph = construct_graph(
            algorithm="empty",
            positions=same_params["positions"],
            features=same_params["features"],
            labels=same_params["labels"],
        )
        degree, _ = calc_deg(x=0, graph_func=lambda var: graph)
        return graph, {"degree": degree}
    if constr_method == "radius_delaunay":
        if "radius_parameter" in same_params:
            radius = same_params["radius_parameter"]["radius"]
            del same_params["radius_parameter"]
        else:
            _, radius, degree = construct_graph_by_degree(
                graph_func=lambda x: construct_graph(
                    algorithm="radius", param=x, **same_params
                ),
                desired_avg_degree=desired_avg_degree,
                tolerance=0.2,
                x0=1 / len(same_params["positions"]),
                verbose=verbose,
            )

        graph = construct_graph(
            algorithm="radius_delaunay", param=radius, **same_params
        )
        degree, _ = calc_deg(x=0, graph_func=lambda var: graph)
        return graph, {"radius": radius, "degree": degree}
    raise ValueError(f"The construction method {constr_method} is not implemented.")


def calc_deg(
    x: float, graph_func: Callable, verbose: bool = False
) -> tuple[float, Data]:
    """Helper function that computes the degree for the given `x`
    using the `graph_func` and the desired degree.
    Used for `construct_graph_by_degree`.

    Args:
        x: The parameter to optimize
        graph_func: The function that computes and returns a graph.
        verbose: If status updates should be given via the console.

    Returns:
        The difference between the desired and the current average degree.
    """
    graph = graph_func(x)
    deg = graph.num_edges / graph.num_nodes
    if verbose:
        print(f"Tested parameter: {x:.2e}\tAverage degree: {deg}")
    return deg, graph


def construct_graph_by_degree(
    graph_func: Callable,
    desired_avg_degree: float = 7.0,
    tolerance: float = 0.5,
    x0: float = 1e-7,
    stopping: int = 15,
    verbose: bool = True,
) -> tuple[Data, float, float]:
    """Construct a graph with the desired average degree by optimizing a parameter.
    Can be used for the `radius` of the radius graph.
    Other usages may be possible but it only works when aiming
    for a smaller degree also means using a smaller parameter.

    Args:
        graph_func: The function that constructs a graph with one argument
            that should be optimized.
        desired_avg_degree: The desired average degree. Defaults to 7.
        tolerance: How close the actual degree should be to the desired degree.
        x0: Initial value for the optimization.
        stopping: To avoid infinite loops, the algorithm stops
            when the difference between actual and desired degree
            doesn't stop after `stopping` times.
        verbose: If status updates should be given via the console.

    Returns:
        The graph, the optimized parameter and the degree as tuple.
    """
    if verbose:
        print(
            f"Searching for a fitting parameter to get an\
                average degree of {desired_avg_degree}:"
        )
    x = x0
    deg, graph = calc_deg(x=x, graph_func=graph_func, verbose=verbose)
    deg_diff = deg - desired_avg_degree
    # Find the right order in which the desired degree lies
    if deg_diff > 0:
        # If the value is positive, the degree of the constructed graph
        # is too large and the parameter should become smaller.
        while deg_diff > 0:
            x /= 10
            deg, graph = calc_deg(x=x, graph_func=graph_func, verbose=verbose)
            deg_diff = deg - desired_avg_degree
        lower_bound, upper_bound = x, x * 10
    elif deg_diff < 0:
        # The oposite of the above when the degree is too small
        while deg_diff < 0:
            x *= 10
            deg, graph = calc_deg(x=x, graph_func=graph_func, verbose=verbose)
            deg_diff = deg - desired_avg_degree
        lower_bound, upper_bound = x / 10, x
    else:
        # Quite unlikely but if we find the perfect degree directly,
        # the algorithm will stop
        return graph, x, deg

    # Use a binary search style optimization:
    # Divide by two until the stopping criterion is reached.
    last_abs_deg_diff = np.abs(deg_diff)
    stopping_count = 0
    while np.abs(deg_diff) - tolerance > 0:
        # Condition to avoid infinite loop
        if last_abs_deg_diff > np.abs(deg_diff):
            stopping_count = 0
        else:
            stopping_count += 1
            if stopping <= stopping_count:
                break
        last_abs_deg_diff = np.abs(deg_diff)

        x = np.mean([lower_bound, upper_bound])  # type: ignore
        deg, graph = calc_deg(x=x, graph_func=graph_func, verbose=verbose)
        deg_diff = deg - desired_avg_degree
        if deg_diff > 0:
            upper_bound = x
        else:
            lower_bound = x

    if verbose:
        print(f"Suitable parameter found: {x:2e}")
    return graph, x, deg
