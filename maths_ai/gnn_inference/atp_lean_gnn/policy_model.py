"""Shared complete-action operations for supervised and actor-critic models."""

import torch
from torch import nn
from .actor_critic_types import ActionSample


class GraphActionPolicy(nn.Module):
    def _validate_pools(self, data, pools):
        if len(pools) != len(data.ptr) - 1:
            raise ValueError("One pool is required per graph.")
        for row, pool in enumerate(pools):
            if pool.artifact_fingerprint != self.retrieval_binding:
                raise ValueError("Action support and model retrieval artifact differ.")
            if pool.local_ids and max(pool.local_ids) >= int(
                data.ptr[row + 1] - data.ptr[row]
            ):
                raise ValueError("Local candidate is outside its graph.")

    def act(self, data, *, candidate_pools, tactic_mask=None, greedy=False):
        self._validate_pools(data, candidate_pools)
        nodes, states, logits, values = self.encode(data, tactic_mask=tactic_mask)
        if len(candidate_pools) != len(states):
            raise ValueError("One candidate pool is required for each graph.")
        dist = torch.distributions.Categorical(logits=logits)
        tactics = logits.argmax(-1) if greedy else dist.sample()
        results = self.decoder.run_many(
            [
                (
                    states[i],
                    nodes[int(data.ptr[i]) : int(data.ptr[i + 1])],
                    tactics[i],
                    pool,
                )
                for i, pool in enumerate(candidate_pools)
            ],
            greedy=greedy,
        )
        return ActionSample(
            tactics,
            dist.log_prob(tactics),
            dist.entropy(),
            tuple(x.arguments for x in results),
            torch.stack([x.count_logp for x in results]),
            torch.stack([x.source_logp for x in results]),
            torch.stack([x.identity_logp for x in results]),
            values.squeeze(-1),
            logits,
        )

    def evaluate_actions(
        self, data, tactic_ids, arguments, *, candidate_pools, tactic_mask=None
    ):
        self._validate_pools(data, candidate_pools)
        nodes, states, logits, values = self.encode(data, tactic_mask=tactic_mask)
        if len(arguments) != len(states) or len(candidate_pools) != len(states):
            raise ValueError(
                "Actions, pools, and graphs must have identical batch lengths."
            )
        results = self.decoder.run_many(
            [
                (
                    states[i],
                    nodes[int(data.ptr[i]) : int(data.ptr[i + 1])],
                    tactic_ids[i],
                    pool,
                )
                for i, pool in enumerate(candidate_pools)
            ],
            forced_rows=arguments,
        )
        return ActionSample(
            tactic_ids,
            torch.distributions.Categorical(logits=logits).log_prob(tactic_ids),
            torch.distributions.Categorical(logits=logits).entropy(),
            tuple(arguments),
            torch.stack([x.count_logp for x in results]),
            torch.stack([x.source_logp for x in results]),
            torch.stack([x.identity_logp for x in results]),
            values.squeeze(-1),
            logits,
        )

    def forward(
        self, data, *, candidate_pools, teacher_tactic_ids=None, teacher_arguments=None
    ):
        if teacher_tactic_ids is not None:
            if teacher_arguments is None:
                raise ValueError(
                    "Teacher-forced actions require complete argument references."
                )
            return self.evaluate_actions(
                data,
                teacher_tactic_ids,
                teacher_arguments,
                candidate_pools=candidate_pools,
            )
        return self.act(data, candidate_pools=candidate_pools, greedy=True)
