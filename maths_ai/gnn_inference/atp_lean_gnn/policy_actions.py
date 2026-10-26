"""Immutable candidate support and complete actions shared by collection/update."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class ArgumentReference:
    source: str
    identity: int

    def __post_init__(self):
        if self.source not in {"local", "library"} or type(self.identity) is not int or self.identity < 0:
            raise ValueError("Argument needs a local/library source and nonnegative integer identity.")


@dataclass(frozen=True)
class CandidatePoolSnapshot:
    local_ids: tuple[int, ...]
    lemma_ids: tuple[int, ...] = ()
    lemma_scores: tuple[float, ...] = ()
    graph_fingerprint: str = ""
    query_fingerprint: str = ""
    artifact_fingerprint: str = ""
    policy_fingerprint: str = ""
    library: object = None
    local_type_signal: tuple[float, ...] = ()

    def __post_init__(self):
        for values in (self.local_ids, self.lemma_ids):
            if type(values) is not tuple or len(set(values)) != len(values) or any(type(x) is not int or x < 0 for x in values):
                raise ValueError("Candidate IDs must be unique immutable nonnegative integers.")
        if len(self.lemma_ids) != len(self.lemma_scores) or any(not math.isfinite(x) for x in self.lemma_scores):
            raise ValueError("Each library candidate needs a finite retrieval score.")
        if self.lemma_ids and (self.library is None or self.library.fingerprint != self.artifact_fingerprint):
            raise ValueError("Library candidates require their exact immutable artifact.")
        if self.local_type_signal and (len(self.local_type_signal) != len(self.local_ids) or any(not math.isfinite(x) for x in self.local_type_signal)):
            raise ValueError("Local type features do not match the pool.")

    @property
    def fingerprint(self):
        return fingerprint({k: v for k, v in self.__dict__.items() if k != "library"})

    def validate(self, action):
        for ref in action:
            if ref.identity not in (self.local_ids if ref.source == "local" else self.lemma_ids):
                raise ValueError("Stored action is outside its collection candidate support.")


def local_pools(data) -> tuple[CandidatePoolSnapshot, ...]:
    """Explicit local-candidate provider for local-only supervised datasets."""
    mask = data.premise_mask.bool()
    return tuple(CandidatePoolSnapshot(tuple((mask[start:end].nonzero().flatten()).tolist()))
                 for start, end in zip(data.ptr[:-1].tolist(), data.ptr[1:].tolist()))
