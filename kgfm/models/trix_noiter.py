"""TRIXNoIter: TRIX with the iterative entity-relation update removed.

See `trix_noiter.md` for the full rationale.

Forward sequence (entity → relation → score), each sub-network runs once:
    1. `entity_model_mini` over the entity graph, with a constant `torch.ones`
       bootstrap for relation_representations. Even with uninformative initial
       relation queries, the entity NBFNet's message passing extracts entity-graph
       structural features into per-entity representations.
    2. `relation_model.node_mlp(...).reshape(B, num_rel_nodes, -1)` projects
       those entity features into the per-relation `node_representations`
       expected by the relation graph's convs — same projection RelNet uses
       internally in TRIX, just hoisted out.
    3. `_RelNetNoMid` runs RelNet's 4 conv stacks (hh/ht/th/tt) with that
       fixed `node_representations`, no internal entity update.
    4. `entity_model` (ULTRA's `EntityNBFNet`, via `_ExternalRelEntityNBFNet`)
       scores using the relation reps from step 3.

Vs TRIX (iterative): same components, same shapes, same entity-aware relation
graph. Only the *order* of sub-calls changes — entity_model_mini moves from
inside RelNet's layer loop to before it. No `relation 1 → entity 1 → relation 1
→ entity 3` ping-pong.

Vs ULTRA: ULTRA seeds relation reps from a learnable `nn.Embedding`, fully
relation-only, no entity context. TRIXNoIter keeps the entity-graph signal
through the bootstrap entity_model_mini pass.
"""

import torch
from torch import nn

from kgfm.models.trix_entity import RelNet, EntityNet
from kgfm.models.ultra import EntityNBFNet


class _ExternalRelEntityNBFNet(EntityNBFNet):
    """EntityNBFNet variant that takes pre-computed relation features
    instead of dispatching to an internal relation_model. Used by TRIXNoIter
    because TRIX's RelNet has a different signature than ULTRA's
    RelNBFNet / MOTIF's RelHCNet (it needs `data` and `batch` and a
    callable mid-pass entity model, not a single rel-graph object).

    Body mirrors `EntityNBFNet.forward` (kgfm/models/ultra.py:163-214)
    minus the relation_model dispatch and the relation-graph rebuild.
    The relation-graph rebuild is skipped because TRIXNoIter's relation
    encoder runs *before* `remove_easy_edges` / `drop_edge_rate` — same
    temporal ordering as `TRIXEntity.forward`.
    """

    def forward(self, data, batch, external_relation_representations):
        h_index, t_index, r_index = batch.unbind(-1)
        shape = h_index.shape

        if self.training and not self.synthetic:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            if self.drop_edge_rate > 0:
                drop_edge_mask = torch.bernoulli(
                    (1 - self.drop_edge_rate)
                    * torch.ones(len(data.edge_type), device=h_index.device)
                ).to(bool)
                data.edge_index = data.edge_index[:, drop_edge_mask]
                data.edge_type = data.edge_type[drop_edge_mask]

        if not self.synthetic:
            h_index, t_index, r_index = self.negative_sample_to_tail(
                h_index, t_index, r_index,
                num_direct_rel=data.num_relations // 2,
            )
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        relation_representations = external_relation_representations
        self.query = relation_representations
        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class _RelNetNoMid(RelNet):
    """RelNet with the i==0 mid-pass removed. Takes pre-computed
    `node_representations` directly instead of an `entity_model_1` callable.

    Body mirrors `RelNet.forward` (kgfm/models/trix_entity.py:90-142) minus
    the `if i == 0:` block; everything else (boundary, edge_weights, the four
    layer stacks, short_cut, concat_hidden tail) is identical.
    """

    def forward(self, data, batch, node_representations):
        rel_graph = data.relation_adj
        h_index = batch[:, 0, 2]

        batch_size = len(h_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(batch_size, rel_graph["hh"].num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (rel_graph["hh"].num_nodes, rel_graph["hh"].num_nodes)
        edge_weight_hh = torch.ones(rel_graph["hh"].num_edges, device=h_index.device)
        edge_weight_ht = torch.ones(rel_graph["ht"].num_edges, device=h_index.device)
        edge_weight_th = torch.ones(rel_graph["th"].num_edges, device=h_index.device)
        edge_weight_tt = torch.ones(rel_graph["tt"].num_edges, device=h_index.device)

        hiddens = []
        layer_input = boundary

        for i in range(len(self.layers_hh)):
            self.layers_hh[i].relation = node_representations
            self.layers_ht[i].relation = node_representations
            self.layers_th[i].relation = node_representations
            self.layers_tt[i].relation = node_representations

            hidden_hh = self.layers_hh[i](layer_input, query, boundary, rel_graph["hh"].edge_index, rel_graph["hh"].edge_type, size, edge_weight_hh)
            hidden_ht = self.layers_ht[i](layer_input, query, boundary, rel_graph["ht"].edge_index, rel_graph["ht"].edge_type, size, edge_weight_ht)
            hidden_th = self.layers_th[i](layer_input, query, boundary, rel_graph["th"].edge_index, rel_graph["th"].edge_type, size, edge_weight_th)
            hidden_tt = self.layers_tt[i](layer_input, query, boundary, rel_graph["tt"].edge_index, rel_graph["tt"].edge_type, size, edge_weight_tt)

            hidden = hidden_hh + hidden_ht + hidden_th + hidden_tt
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, rel_graph["hh"].num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return output


class TRIXNoIter(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_1_cfg, entity_model_cfg):
        super(TRIXNoIter, self).__init__()

        self.relation_model = _RelNetNoMid(**rel_model_cfg)
        self.entity_model_mini = EntityNet(**entity_model_1_cfg)
        self.entity_model = _ExternalRelEntityNBFNet(**entity_model_cfg)

    def forward(self, data, batch, precomputed_rel_emb=None):
        # precomputed_rel_emb is accepted for API parity with run.py / run_many.py
        # (see kgfm/models/motif.py:25). TRIXNoIter computes its own per-batch
        # relation features, so feeding it here is a no-op — assert it isn't
        # accidentally provided.
        assert precomputed_rel_emb is None, \
            "TRIXNoIter does not support precomputed_rel_emb (relation features are computed per-batch)"
        rel_graph = data.relation_adj
        h_index_first = batch[:, 0, 2]
        batch_size = len(h_index_first)
        num_rel_nodes = rel_graph["hh"].num_relations
        dim = self.relation_model.layers_hh[0].input_dim

        # Step 1: bootstrap rel reps with torch.ones — uninformative as relation queries,
        # but entity_model_mini still extracts entity-graph structure via message passing.
        bootstrap_rel_reps = torch.ones(
            batch_size, num_rel_nodes, dim, device=h_index_first.device
        )

        # Step 2: entity-pass — entity-aware features from one bellmanford over the entity graph.
        entity_features = self.entity_model_mini(data, bootstrap_rel_reps, batch)["feature"]

        # Step 3: project entity features → per-relation node_representations.
        # Same projection RelNet uses internally (node_mlp + reshape to [B, num_rel_nodes, -1]).
        node_representations = self.relation_model.node_mlp(entity_features).reshape(
            batch_size, num_rel_nodes, -1
        )

        # Step 4: relation-pass with fixed node_representations (no mid-pass loop).
        relation_representations = self.relation_model(data, batch, node_representations)

        # Step 5: entity-score.
        score = self.entity_model(
            data, batch,
            external_relation_representations=relation_representations,
        )
        return score
