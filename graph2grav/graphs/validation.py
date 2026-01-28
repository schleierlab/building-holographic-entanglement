"""Graph and node validation using Pydantic models.

This module provides validation for graph structures to ensure they have
the required attributes for use in graph2grav computations.
"""

import networkx as nx
from typing import Annotated, Optional, Literal
from pydantic import BaseModel, AfterValidator, model_validator, ValidationError


# Required attributes for validation
required_graph_labels = ["number_of_boundaries", "periodic"]
required_node_labels = ["is_ancilla", "is_boundary"]


# --- Validator Functions ---

def ensure_true(value: bool) -> bool:
    """
    Ensure that the value is True.
    """
    if not value:
        raise ValueError("Value must be True")
    return value

def ensure_false(value: bool) -> bool:
    """
    Ensure that the value is False.
    """
    if value:
        raise ValueError("Value must be False")
    return value


# --- Pydantic Models ---

class Graph(BaseModel):
    """
    A graph with the following attributes:
    - number_of_boundaries: int
    - periodic: bool
    """

    number_of_boundaries: int
    periodic: bool
    subdivided: bool

class Node(BaseModel):
    """
    A node with the following attributes:
    - is_ancilla: bool
    - is_boundary: bool
    """
    graph: Graph
    region: Optional[Literal["bulk", "probe"]] = None
    tier: Optional[Literal["primary", "secondary"]] = None

    @model_validator(mode="after")
    def enforce_region_and_tiers(self):
        if self.graph.subdivided and self.region is None:
            raise ValueError("Region is required when graph is subdivided")
        if self.graph.subdivided and self.tier is None:
            raise ValueError("Tier is required when graph is subdivided")

        return self


class BoundaryNode(Node):
    """
    A node that is a boundary node with the following attributes:
    - is_ancilla: bool
    - is_boundary: bool
    - boundary_index: int
    - position_in_boundary: int
    """

    is_ancilla: Annotated[bool, AfterValidator(ensure_false)]
    is_boundary: Annotated[bool, AfterValidator(ensure_true)]
    boundary_index: int
    position_in_boundary: int
    b0: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b1: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b2: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b3: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b4: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b5: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b6: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b7: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b8: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None
    b9: Optional[Annotated[bool, AfterValidator(ensure_true)]] = None



    @model_validator(mode="after")
    def check_for_boundary_nickname(self):
        if not self.__dict__[f"b{self.boundary_index}"]:
            raise ValueError(f"boundary nickname 'b{self.boundary_index}' should be True")

        for i in range(self.graph.number_of_boundaries):
            if i == self.boundary_index:
                continue


            if f"b{i}" not in self.__dict__:
                raise ValueError(f"boundary nickname 'b{i}' should exist and be set to False.")
            elif self.__dict__[f"b{i}"]:
                raise ValueError(f"boundary nickname 'b{i}' should be False")

        return self

    @model_validator(mode="after")
    def less_than_number_of_boundaries(self):
        """
        Ensure that the boundary index is less than the number of boundaries.
        """
        if self.boundary_index >= self.graph.number_of_boundaries:
            raise ValueError(f"Boundary index {self.boundary_index} must be less than number of boundaries {self.graph.number_of_boundaries}")

        return self


class AncillaNode(Node):
    """
    A node that is an ancilla node with the following attributes:
    - is_ancilla: bool
    - is_boundary: bool
    """
    is_ancilla: Annotated[bool, AfterValidator(ensure_true)]
    is_boundary: Annotated[bool, AfterValidator(ensure_false)]
    b0: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b1: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b2: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b3: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b4: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b5: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b6: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b7: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b8: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None
    b9: Optional[Annotated[bool, AfterValidator(ensure_false)]] = None

    @model_validator(mode="after")
    def check_for_boundary_nickname(self):

        for i in range(self.graph.number_of_boundaries):
            if self.__dict__[f"b{i}"] is None or self.__dict__[f"b{i}"]:
                raise ValueError(f"boundary nickname 'b{i}' should exist and be set to False.")

        return self




def check_labels(G: nx.Graph):
    """
    Check the labels of the nodes in a graph using pydantic-based validation.
    """

    for node in G.nodes:
        try:
            if G.nodes[node]["is_ancilla"]:
                AncillaNode.model_validate(G.nodes[node])
            else:
                BoundaryNode.model_validate(G.nodes[node])
        except ValidationError as e:
            print(f"Node {node} in graph fails validation.")
            print("Node information:", G.nodes[node])
            raise e

    return True
