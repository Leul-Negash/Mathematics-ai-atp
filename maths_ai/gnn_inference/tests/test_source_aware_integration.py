import torch
from torch_geometric.data import Data, Batch
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec
from maths_ai.gnn_inference.atp_lean_gnn.model_factory import build_actor_critic_model
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import local_pools
from maths_ai.gnn_inference.atp_lean_gnn.checkpointing import checkpoint_payload, build_model_from_checkpoint
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC


def fixture():
    spec = ModelSpec.from_dict(dict(architecture='gatv2', hidden_dim=8, dropout=.5,
                                    encoder=dict(heads=2, num_layers=1, readout='state_mean_attention'), text_dim=4))
    model = build_actor_critic_model(model_spec=spec, num_node_labels=5, num_tactics=3)
    data = Data(x=torch.tensor([1,2,3]), node_type=torch.tensor([0,0,0]),
                edge_index=torch.tensor([[0,1,2],[1,2,0]]), premise_mask=torch.tensor([True,False,True]), state_node_index=torch.tensor([2]))
    batch=Batch.from_data_list([data,data])
    return model,batch,local_pools(batch)


def test_dropout_disabled_recomputation_preserves_full_policy_and_gradients():
    model,batch,pools=fixture()
    model.train()
    with torch.no_grad():
        sampled=model.act(batch,candidate_pools=pools)
    model.train()
    evaluated=model.evaluate_actions(batch,sampled.tactic_action,sampled.arguments,candidate_pools=pools)
    torch.testing.assert_close(sampled.logp,evaluated.logp,rtol=0,atol=0)
    (-evaluated.logp.mean()+evaluated.value.square().mean()).backward()
    assert sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)>0
    assert sum(p.grad.abs().sum() for p in model.critic.parameters() if p.grad is not None)>0


def test_version_four_roundtrip_has_identical_actions():
    model,batch,pools=fixture()
    nodes={str(i):i for i in range(5)};tactics={str(i):i for i in range(3)}
    payload=checkpoint_payload(model_kind='actor_critic_with_args',model_spec=model.model_spec,node_vocab=nodes,
        tactic_vocab=tactics,model=model,graph_representation=MODEL_SEXPR_GRAPH_SPEC)
    assert payload['manifest']['checkpoint_format_version']==4
    loaded,_,_=build_model_from_checkpoint(payload,node_vocab=nodes,tactic_vocab=tactics,graph_representation=MODEL_SEXPR_GRAPH_SPEC)
    first=model.act(batch,candidate_pools=pools,greedy=True)
    second=loaded.act(batch,candidate_pools=pools,greedy=True)
    assert first.arguments==second.arguments
    torch.testing.assert_close(first.logp,second.logp)
