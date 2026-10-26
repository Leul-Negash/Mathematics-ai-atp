"""Supervised model uses the same source-aware policy as actor-critic collection."""
import torch
from torch import nn
from .policy_model import GraphActionPolicy
from .source_aware_decoder import SourceAwareDecoder
from .policy_actions import ArgumentReference, local_pools

class TacticWithArgsClassifier(GraphActionPolicy):
    def __init__(self, *, encoder, num_tactics, dropout=0.2, max_args=3, text_dim=384, local_type_bonus_strength=0.0):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim, self.max_args = encoder.output_dim, max_args
        self.tactic_classifier = nn.Linear(self.hidden_dim, num_tactics)
        self.tactic_dropout = nn.Dropout(dropout)
        self.decoder = SourceAwareDecoder(self.hidden_dim, num_tactics, max_args, text_dim, local_type_bonus_strength)

    def encode_graph(self, data):
        self.eval()
        return self.encoder(data)

    def predict_tactics(self, encoded, *, tactic_mask=None):
        logits = self.tactic_classifier(encoded.state_embeddings)
        return logits if tactic_mask is None else logits.masked_fill(~tactic_mask, -torch.inf)

    def encode(self, data, *, tactic_mask=None):
        encoded = self.encode_graph(data)
        return encoded.node_embeddings, encoded.state_embeddings, self.predict_tactics(encoded, tactic_mask=tactic_mask), encoded.state_embeddings.sum(-1, keepdim=True) * 0


def supervised_actions(batch, max_args):
    if not hasattr(batch, 'arg_count') or not hasattr(batch, 'arg_node_indices'):
        raise ValueError("Supervised argument training requires explicit argument targets.")
    rows=[]
    offset=0
    lemma_ids=getattr(batch, 'arg_lemma_ids', torch.full_like(batch.arg_node_indices, -1))
    for count in batch.arg_count.tolist():
        if not 0 <= count <= max_args:
            raise ValueError("Argument target exceeds the policy budget.")
        args=[]
        for local,lemma in zip(batch.arg_node_indices[offset:offset+count].tolist(), lemma_ids[offset:offset+count].tolist()):
            if (local>=0)==(lemma>=0):
                raise ValueError("Each argument needs exactly one source.")
            args.append(ArgumentReference('local' if local>=0 else 'library', int(local if local>=0 else lemma)))
        rows.append(tuple(args));offset+=count
    return tuple(rows)


def supervised_loss(model, batch, *, candidate_pools, unknown_tactic_id=0):
    actions=supervised_actions(batch, model.max_args)
    result=model.evaluate_actions(batch,batch.y.view(-1),actions,candidate_pools=candidate_pools)
    valid=batch.y.view(-1)!=unknown_tactic_id
    loss=-result.logp[valid].mean() if valid.any() else result.logp.sum()*0
    return loss, {'tactic_loss': float(-result.tactic_logp[valid].mean().detach()) if valid.any() else 0.,
                  'arg_loss': float(-result.arg_logp[valid].mean().detach()) if valid.any() else 0., 'total_loss': float(loss.detach())}, result
