"""MOTIF model + RelHCNet (extracted from MOTIF/motif/models.py).

`EntityNBFNet` is shared with ULTRA — imported from `.ultra` rather than
duplicated. Module attribute names (`relation_model`, `entity_model`, `layers`,
`mlp`, ...) preserved verbatim.
"""

import torch
from torch import nn
from torch.nn import functional as F

from kgfm import layers
from kgfm.util import static_positional_encoding
from kgfm.models.ultra import EntityNBFNet


class MOTIF(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(MOTIF, self).__init__()

        self.relation_model = RelHCNet(num_relation=7, **rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

    def forward(self, data, batch, precomputed_rel_emb=None):
        score = self.entity_model(data, batch, self.relation_model,
                                  relation_hyper_flag=True,
                                  precomputed_rel_emb=precomputed_rel_emb)
        return score


class RelHCNet(nn.Module):
    def __init__(self, input_dim, num_relation, hidden_dims,
                 short_cut=True, num_mlp_layer=2, max_arity=3, dropout=0.2,
                 norm="layer_norm", padding_idx=0, dependent=False,
                 aggregate_func="sum", drop_edge_rate=0.0, **kwargs):
        super(RelHCNet, self).__init__()
        self.name = "RelHCNet"
        self.aggregate_func = aggregate_func
        self.drop_edge_rate = drop_edge_rate
        assert self.drop_edge_rate >= 0.0 and self.drop_edge_rate < 1.0
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut

        self.max_arity = max_arity
        self.padding_idx = padding_idx

        self.max_considered_arity = kwargs.get("max_considered_arity", 2)

        static_encodings = static_positional_encoding(max_arity + 1, input_dim)
        self.position = nn.Embedding.from_pretrained(static_encodings, freeze=True)
        self.position.weight.data[self.padding_idx] = torch.ones(input_dim)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.HypergraphLayer(self.dims[i], self.dims[i + 1], self.num_relation,
                                                      dropout=dropout, norm=norm, dependent=dependent,
                                                      aggregate_func=aggregate_func))

        self.feature_dim = input_dim

        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(self.feature_dim, self.feature_dim))
        self.mlp = nn.Sequential(*mlp)

    def inference(self, query_idx, edge_list, rel_list, num_nodes):
        batch_size = len(query_idx)

        query = torch.ones(query_idx.shape[0], self.dims[0], device=query_idx.device, dtype=torch.float)
        index = query_idx.unsqueeze(-1).expand_as(query)
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=query_idx.device)

        query_feature.scatter_add_(dim=1,
                                   index=index.unsqueeze(1),
                                   src=query.unsqueeze(1))

        init_feature = query_feature
        init_feature[:, self.padding_idx, :] = 0

        layer_input = init_feature

        for layer in self.layers:
            hidden = F.relu(layer(layer_input, query, edge_list, rel_list))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden
        output = layer_input

        score = self.mlp(output)

        return score

    def _inference_sharded(self, query_idx, hypergraph_shards, num_nodes):
        batch_size = len(query_idx)

        query = torch.ones(query_idx.shape[0], self.dims[0], device=query_idx.device, dtype=torch.float)
        index = query_idx.unsqueeze(-1).expand_as(query)
        query_feature = torch.zeros(batch_size, num_nodes, self.dims[0], device=query_idx.device)

        query_feature.scatter_add_(dim=1,
                                   index=index.unsqueeze(1),
                                   src=query.unsqueeze(1))

        init_feature = query_feature
        init_feature[:, self.padding_idx, :] = 0

        layer_input = init_feature
        for layer in self.layers:
            hidden = F.relu(layer.forward_sharded(layer_input, query, hypergraph_shards))
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return self.mlp(layer_input)

    def forward(self, relation_hypergraph, query):
        from kgfm.motif_aristo_shim import HypergraphShards
        if isinstance(relation_hypergraph, HypergraphShards):
            assert not self.training or self.drop_edge_rate == 0.0, \
                "drop_edge_rate not supported on the sharded path; AristoV4-MOTIF is inference-only"
            num_nodes_padded = relation_hypergraph.num_nodes + 1
            query_shifted = query + 1
            relation_feature = self._inference_sharded(query_shifted, relation_hypergraph, num_nodes_padded)
            return relation_feature[:, 1:, :]

        edge_list, rel_list, num_nodes = relation_hypergraph.edge_index, relation_hypergraph.edge_type, relation_hypergraph.num_nodes

        if self.training and self.drop_edge_rate >= 0:
            drop_edge_mask = torch.bernoulli((1 - self.drop_edge_rate) * torch.ones(len(rel_list), device=edge_list.device)).to(bool)
            edge_list = edge_list[:, drop_edge_mask]
            rel_list = rel_list[drop_edge_mask]

        # Shift edge_list by 1 to avoid the padding node
        edge_list = torch.transpose(edge_list, 0, 1) + 1
        query += 1
        num_nodes += 1

        relation_feature = self.inference(query, edge_list, rel_list, num_nodes)

        query -= 1
        return relation_feature[:, 1:, :]
