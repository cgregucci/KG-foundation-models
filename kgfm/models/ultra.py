"""ULTRA model + helpers (extracted from MOTIF/motif/models.py).

Module attribute names (`relation_model`, `entity_model`, `layers`, `mlp`, ...)
are preserved verbatim from upstream so that pretrained checkpoints
(`ckpts/ultra_3g.pth`, `ckpts/motif_3g.pth`) load unchanged.
"""

import torch
from torch import nn
from torch.nn import functional as F

from kgfm import layers
from kgfm.base_nbfnet import BaseNBFNet


class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(Ultra, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

    def forward(self, data, batch, precomputed_rel_emb=None):
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs

        score = self.entity_model(data, batch, self.relation_model,
                                  relation_hyper_flag=False,
                                  precomputed_rel_emb=precomputed_rel_emb)
        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
            )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]
        return output


class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, drop_edge_rate=0, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)
        assert drop_edge_rate >= 0 and drop_edge_rate < 1
        self.drop_edge_rate = drop_edge_rate
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

        self.synthetic = kwargs.get("synthetic", False)

    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, batch, relation_model, relation_hyper_flag=False,
                precomputed_rel_emb=None):
        from kgfm import tasks  # local import to avoid cycle when models is imported by tasks transitively

        h_index, t_index, r_index = batch.unbind(-1)
        shape = h_index.shape

        if self.training and not self.synthetic:
            assert precomputed_rel_emb is None, \
                "precomputed_rel_emb is only supported in eval mode (training rebuilds the relation graph per batch)"
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            if self.drop_edge_rate > 0:
                drop_edge_mask = torch.bernoulli((1 - self.drop_edge_rate) * torch.ones(len(data.edge_type), device=h_index.device)).to(bool)
                data.edge_index, data.edge_type = data.edge_index[:, drop_edge_mask], data.edge_type[drop_edge_mask]

            if relation_hyper_flag:
                data = tasks.build_relation_hypergraph(data)
            else:
                data = tasks.build_relation_graph(data)

        if not self.synthetic:
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        if precomputed_rel_emb is not None:
            # Skip the relation_model call entirely. The precomputed table has
            # shape [R, R, D] (one row per query relation); indexing by the
            # per-batch query gives the same [B, R, D] tensor the call would
            # have produced.
            relation_representations = precomputed_rel_emb[r_index[:, 0]]
        elif relation_hyper_flag:
            if not self.synthetic:
                relation_representations = relation_model(data.relation_hypergraph, query=r_index[:, 0])
            else:
                index_to_use = relation_model.max_considered_arity - 2
                relation_representations = relation_model(data.relation_hypergraph[0][index_to_use], query=r_index[:, 0])
        else:
            relation_representations = relation_model(data.relation_graph, query=r_index[:, 0])

        self.query = relation_representations

        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
