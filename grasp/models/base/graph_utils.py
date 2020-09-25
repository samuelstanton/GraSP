import os
import math
import torch
from graphviz import Digraph

class GraphNode(object):
    """Simple aggregation op"""
    def __init__(self, in_degree):
        super().__init__()
        self.in_degree = in_degree

    def __call__(self, edge_features):
        return torch.stack(edge_features).sum(0)


class GraphEdge(torch.nn.Module):
    """Module for computing weighted outputs for an arbitrary collection of ops"""
    def __init__(self, op_dict, op_weights=None):
        super().__init__()
        self.op_dict = torch.nn.ModuleDict(op_dict)
        if op_weights is None:
            op_weights = torch.ones(self.num_ops) / self.num_ops
        else:
            assert op_weights.size(0) == len(op_dict)
        self.register_parameter('weight', torch.nn.Parameter(op_weights))

    def forward(self, node_features):
        edge_features = torch.stack([
            weight * op(node_features) for weight, op in zip(self.weight.exp(), self.op_modules)
        ])
        return edge_features.sum(0)

    def remove_ops(self, idxs):
        keys = list(self.op_dict.keys())
        for idx in idxs:
            del self.op_dict[keys[idx]]

    @property
    def num_ops(self):
        return len(self.op_dict)

    @property
    def op_names(self):
        return list(self.op_dict.keys())

    @property
    def op_modules(self):
        return list(self.op_dict.values())


class GraphLayer(torch.nn.Module):
    """
        Module for representing an arbitrary computation graph. Ops can be masked and removed
        using the `sparsify` method. The graph structure is defined by `adj_dict`, which has items of
        the form `{tail_node_idx: (head_node_idx, edge_idx)}`. The total number of tuples should equal the
        number of edges. `adj_dict` can be thought of as a compact representation of the adjacency matrix.
        The keys of `adj_dict` should go from `1` to `N-1`, where `N` is the number of nodes in the graph. The
        first node is defined implicitly.
    """

    def __init__(self, edges, adj_dict):
        """
            edges: list of op modules,
            adj_dict: dict defining edge adjacency
        """
        super().__init__()
        self.edges = torch.nn.ModuleList(edges)
        self.adj_dict = adj_dict
        self.nodes = self._create_nodes()
        self._init_op_weights()

    def _create_nodes(self):
        nodes = [GraphNode(in_degree=1)]
        for tail_node_id, head_edge_tuples in self.adj_dict.items():
            in_degree = sum([self.edges[edge_id].num_ops for _, edge_id in head_edge_tuples])
            nodes.append(GraphNode(in_degree))
        return nodes

    def _init_op_weights(self):
        for tail_node_idx, head_edge_tuple in self.adj_dict.items():
            tail_node = self.nodes[tail_node_idx]
            stdv = 1. / math.sqrt(self.num_ops)
            for _, edge_idx in head_edge_tuple:
                edge = self.edges[edge_idx]
                # edge.weight.data.uniform_(-stdv, stdv)
                edge.weight.data.uniform_(-1, 0)
                # edge.weight.data = torch.ones_like(edge.weight) / tail_node.in_degree

    def forward(self, inputs):
        node_features = [self.nodes[0]([inputs])]
        for tail_node_idx, head_edge_tuples in self.adj_dict.items():
            edge_features = []
            for head_node_idx, edge_idx in head_edge_tuples:
                edge_features.append(self.edges[edge_idx](node_features[head_node_idx]))
            if len(edge_features) > 0:
                node_features.append(self.nodes[tail_node_idx](edge_features))
            else:
                node_features.append(torch.zeros_like(inputs) * inputs)
        return node_features[-1]

    def sparsify(self, op_masks):
        assert len(op_masks) == self.num_edges
        for edge_idx, mask in enumerate(op_masks):
            if torch.all(mask == 0):
                continue
            if torch.any(mask == 0):
                drop_idxs = [idx.item() for idx in torch.nonzero(mask == 0, as_tuple=False)]
                self.edges[edge_idx].remove_ops(drop_idxs)
                op_dict = dict(self.edges[edge_idx].op_dict.items())
                op_weights = self.edges[edge_idx].weight[mask]
                self.edges[edge_idx] = GraphEdge(op_dict, op_weights)

        drop_edges = [i for i in range(self.num_edges) if torch.all(op_masks[i] == 0)]
        self.remove_edges(drop_edges)

    def remove_edges(self, idxs):
        # update edge list
        self.edges = torch.nn.ModuleList(drop_list_elements(self.edges, idxs))
        # update adjacency dict
        edge_count = 0
        for tail_node_idx, head_edge_tuples in self.adj_dict.items():
            drop_tuples = []
            for tuple_idx, (head_node_idx, edge_idx) in enumerate(head_edge_tuples):
                if edge_idx in idxs:
                    drop_tuples.append(tuple_idx)
                else:
                    self.adj_dict[tail_node_idx][tuple_idx] = (head_node_idx, edge_count)
                    edge_count += 1
            self.adj_dict[tail_node_idx] = drop_list_elements(self.adj_dict[tail_node_idx], drop_tuples)
        # reconstruct aggregation nodes
        self.nodes = self._create_nodes()

    def draw_graph(self, log_dir, graph_name, view=False):
        graph = Digraph()
        for node_idx in range(self.num_nodes):
            graph.node(str(node_idx))
        for tail_node_idx, head_edge_tuples in self.adj_dict.items():
            for head_node_idx, edge_idx in head_edge_tuples:
                for op_name in self.edges[edge_idx].op_dict.keys():
                    graph.edge(str(head_node_idx), str(tail_node_idx), label=op_name)
        graph.render(os.path.join(log_dir, f"{graph_name}.pdf"), view=view)
        graph.save(graph_name, log_dir)

    @property
    def edge_weights(self):
        edge_weights = torch.stack([edge.weight for edge in self.edges])
        try:
            grads = torch.stack(edge.weight.grad for edge in self.edges)
            edge_weights.grad = grads
        except:
            pass
        return edge_weights

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def num_ops(self):
        num_ops = 0
        for edge in self.edges:
            num_ops += edge.num_ops
        return num_ops


def drop_list_elements(old_list, idxs):
    num_items = len(old_list)
    new_list = [old_list[i] for i in range(num_items) if i not in idxs]
    return new_list
