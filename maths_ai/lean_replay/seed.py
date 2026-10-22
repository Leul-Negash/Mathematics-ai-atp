"""Typed goal seeds and their lowering to one replayable Lean statement.

A seed is an ordered local context plus a target. It comes from printed text
(``seed_from_text``) or from structured raw expressions (``seed_from_records``,
used for dataset rows and for canonical goals the REPL already serialized).
Both lower through ``render_replay`` to a closed ``∀``/``let`` statement, the
``intro`` pattern that recreates the locals, and the universe names to declare.

Inaccessible locals (``inst✝³``) are bound under their base name and introduced
with ``_``, so Lean prints the same display names again. A local is renamed only
when keeping its name would capture a later reference; every rename is listed
in the replay's ``name_map``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

from . import raw as R
from .errors import ReplayFailure, ReplayInputError
from .lexer import Token, display_base, is_inaccessible_display, tokenize
from .text import parse_state
from .universe import level_aliases, level_references

RENDERER_VERSION = "lean-replay-v1"
BINDER_ROLES = frozenset(
    {":explicit", ":implicit", ":strict-implicit", ":instance-implicit", ":let"}
)
_ROLE_BRACKETS = {
    ":explicit": ("(", ")"),
    ":implicit": ("{", "}"),
    ":strict-implicit": ("⦃", "⦄"),
    ":instance-implicit": ("[", "]"),
    None: ("(", ")"),
}
_INFO_BRACKETS = {
    "default": ("(", ")"),
    "implicit": ("{", "}"),
    "strict-implicit": ("⦃", "⦄"),
    "instance": ("[", "]"),
}

SeedExpr = Union[str, R.RawExpr]
SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True)
class SeedLocal:
    """One local declaration of a seed.

    ``name`` is the display name (``inst✝³`` for an inaccessible local).
    ``role`` is ``None`` only for printed text, which does not show it.
    ``internal_id`` and ``context_index`` exist for structured locals.
    """

    name: str
    type: SeedExpr
    value: SeedExpr | None = None
    role: str | None = None
    internal_id: str | None = None
    context_index: int | None = None

    @property
    def inaccessible(self) -> bool:
        return is_inaccessible_display(self.name)

    @property
    def is_let(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class LeanGoalSeed:
    """An unelaborated goal: ordered locals, a target and where it came from.

    ``origin`` is ``text`` or ``structured``. Text seeds hold Lean source
    strings; structured seeds hold decoded raw expressions. ``source`` is
    immutable provenance (dataset revision, split, row, theorem, ...).
    """

    locals: tuple[SeedLocal, ...]
    target: SeedExpr
    origin: str
    source: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in {"text", "structured"}:
            raise ValueError(f"Unknown seed origin {self.origin!r}.")
        expected = str if self.origin == "text" else R.RawExpr.__args__  # type: ignore[attr-defined]
        parts = [self.target, *(local.type for local in self.locals)]
        parts += [local.value for local in self.locals if local.value is not None]
        if any(not isinstance(part, expected) for part in parts):
            raise ValueError(f"A {self.origin} seed mixes expression kinds.")

    @property
    def source_dict(self) -> dict[str, object]:
        return dict(self.source)

    def _raw_parts(self) -> list[R.RawExpr]:
        parts = [self.target, *(local.type for local in self.locals)]
        return parts + [local.value for local in self.locals if local.value is not None]


@dataclass(frozen=True)
class GoalReplaySpec:
    """The Lean input that recreates one proof state.

    ``expression`` is a closed statement over ``universe_levels``; ``local_names``
    is the ``intro`` pattern (``_`` for inaccessible locals) and
    ``expected_names`` the display names Lean should print afterwards.
    ``name_map`` lists only source names the replay had to change.
    """

    expression: str
    local_names: tuple[str, ...]
    universe_levels: tuple[str, ...] = ()
    expected_names: tuple[str, ...] = ()
    name_map: tuple[tuple[str, str], ...] = ()
    level_map: tuple[tuple[str, str], ...] = ()
    fingerprint: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "local_names": list(self.local_names),
            "universe_levels": list(self.universe_levels),
            "expected_names": list(self.expected_names),
            "name_map": [list(pair) for pair in self.name_map],
            "level_map": [list(pair) for pair in self.level_map],
            "fingerprint": self.fingerprint,
            "renderer": RENDERER_VERSION,
        }


# Adapters --------------------------------------------------------------------


def seed_from_text(
    state: str | None = None,
    *,
    goal: str | None = None,
    hypotheses: Sequence[str] = (),
    source: Mapping[str, object] | None = None,
) -> LeanGoalSeed:
    """Build a seed from printed text.

    Pass either a whole printed ``state`` or a ``goal`` with ``hypotheses``,
    each written as a declaration (``n m : Nat``, ``x : Nat := 3``). Both go
    through the same declaration parser; nothing is split on colons by hand.
    """
    if (state is None) == (goal is None):
        raise ValueError("Pass either a printed state or a goal with hypotheses.")
    if state is None:
        lines = [*hypotheses, f"⊢ {goal}"]
        if any("\n" in line for line in lines[:-1]):
            raise ReplayInputError(
                ReplayFailure.MALFORMED_TEXT,
                "A hypothesis entry must be one declaration on one line.",
            )
        state = "\n".join(lines)
    parsed = parse_state(state)
    locals_: list[SeedLocal] = []
    for hypothesis in parsed.hypotheses:
        for name in hypothesis.names:
            locals_.append(
                SeedLocal(name, hypothesis.type_expr, hypothesis.value, hypothesis.role)
            )
    return LeanGoalSeed(
        tuple(locals_),
        parsed.goal,
        "text",
        tuple(sorted((source or {}).items())),
    )


@dataclass(frozen=True)
class StructuredLocal:
    """A local as Pantograph serialized it: identity, role and raw sexps."""

    internal_id: str
    name: str
    context_index: int
    binder_role: str
    is_instance: bool
    is_let: bool
    type_sexp: str
    value_sexp: str | None = None


def seed_from_records(
    target_sexp: str,
    local_records: Sequence[StructuredLocal],
    *,
    source: Mapping[str, object] | None = None,
) -> LeanGoalSeed:
    """Build a seed from raw expressions, checking every reference and role."""
    seen_ids: set[str] = set()
    previous_index = -1
    locals_: list[SeedLocal] = []
    for position, record in enumerate(local_records):
        where = f"local {position} ({record.name})"
        if record.binder_role not in BINDER_ROLES:
            raise _structured(f"Unknown binder role {record.binder_role!r}.", where)
        if record.is_instance != (record.binder_role == ":instance-implicit"):
            raise _structured("is_instance disagrees with the binder role.", where)
        if record.is_let != (record.binder_role == ":let"):
            raise _structured("is_let disagrees with the binder role.", where)
        if not record.internal_id or record.internal_id in seen_ids:
            raise _structured("Local IDs must be present and unique.", where)
        if type(record.context_index) is not int or record.context_index <= previous_index:
            raise _structured("Context indices must increase.", where)
        if not record.name:
            raise _structured("A local needs a display name.", where)
        if record.is_let and not record.value_sexp:
            raise ReplayInputError(
                ReplayFailure.MISSING_CONTEXT,
                "A local definition has no recorded value.",
                field=where,
            )
        if not record.is_let and record.value_sexp:
            raise _structured("Only a local definition carries a value.", where)
        type_ = R.decode_raw(record.type_sexp, where)
        value = R.decode_raw(record.value_sexp, where) if record.is_let else None
        for part in (type_, value):
            if part is None:
                continue
            for fvar in R.free_vars(part):
                if fvar not in seen_ids:
                    raise ReplayInputError(
                        ReplayFailure.MISSING_CONTEXT,
                        f"Reference to {fvar}, which is not an earlier local.",
                        field=where,
                    )
        seen_ids.add(record.internal_id)
        previous_index = record.context_index
        locals_.append(
            SeedLocal(
                record.name,
                type_,
                value,
                record.binder_role,
                record.internal_id,
                record.context_index,
            )
        )
    target = R.decode_raw(target_sexp, "target")
    for fvar in R.free_vars(target):
        if fvar not in seen_ids:
            raise ReplayInputError(
                ReplayFailure.MISSING_CONTEXT,
                f"The target refers to {fvar}, which is not in the context.",
                field="target",
            )
    for part in (target, *(local.type for local in locals_)):
        if R.loose_bvars(part, {}):
            raise _structured("A closed expression has a loose bound variable.", "target")
    return LeanGoalSeed(
        tuple(locals_),
        target,
        "structured",
        tuple(sorted((source or {}).items())),
    )


def _structured(detail: str, where: str) -> ReplayInputError:
    return ReplayInputError(ReplayFailure.MALFORMED_STRUCTURED, detail, field=where)


# Lowering --------------------------------------------------------------------


def _fresh(base: str, used: set[str]) -> str:
    counter = 1
    while f"{base}_{counter}" in used:
        counter += 1
    name = f"{base}_{counter}"
    used.add(name)
    return name


def _plan_names(
    seed: LeanGoalSeed,
    references: list[set[str]],
    reserved: set[str],
) -> list[str]:
    """Choose each local's binder name.

    ``references[k]`` is what part ``k`` refers to: parts ``0..n-1`` are the
    locals' types and values and part ``n`` is the target. A local keeps its
    base name unless a later local rebinds that name and something after the
    rebinding still refers to it, or the name would hide a constant.

    Source syntax cannot name a shadowed local, and finding it by type
    (``‹Ring R›``) leaves tactic metavariables that Mathlib's unifier cannot
    solve within its heartbeat budget, so such a local is renamed.
    """
    locals_ = seed.locals
    bases = [display_base(local.name) for local in locals_]
    used = set(bases) | reserved
    names: list[str] = []
    for i, local in enumerate(locals_):
        key = local.internal_id if local.internal_id is not None else local.name
        shadow = next((j for j in range(i + 1, len(locals_)) if bases[j] == bases[i]), None)
        captured = shadow is not None and any(
            key in references[k] for k in range(shadow + 1, len(references))
        )
        if captured or bases[i] in reserved:
            names.append(_fresh(bases[i], used))
        else:
            names.append(bases[i])
    return names


def _intro_and_expected(seed: LeanGoalSeed, binder_names: list[str]):
    """The intro pattern and the names Lean will print for it.

    Lean shows inaccessible locals sharing a base as ``b✝``, ``b✝¹``, ``b✝²``
    counting back from the newest, so the number depends on how many later
    inaccessible locals keep the same binder name.
    """
    intro, expected, renames = [], [], []
    for i, (local, binder) in enumerate(zip(seed.locals, binder_names)):
        if local.inaccessible:
            intro.append("_")
            later = sum(
                other.inaccessible and other_binder == binder
                for other, other_binder in zip(seed.locals[i + 1 :], binder_names[i + 1 :])
            )
            shown = f"{binder}✝" + (str(later).translate(SUPERSCRIPTS) if later else "")
        else:
            intro.append(binder)
            shown = binder
        expected.append(shown)
        if shown != local.name:
            renames.append((local.name, shown))
    return intro, expected, renames


def _fingerprint(seed: LeanGoalSeed, payload: dict[str, object]) -> str:
    body = {
        "renderer": RENDERER_VERSION,
        "origin": seed.origin,
        "source": [[key, str(value)] for key, value in seed.source],
        **payload,
    }
    encoded = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_replay(seed: LeanGoalSeed) -> GoalReplaySpec:
    if seed.origin == "text":
        return _render_text(seed)
    return _render_structured(seed)


def _wrap(seed: LeanGoalSeed, binder_names, types, values, target: str) -> str:
    expression = target
    for local, name, type_, value in reversed(
        list(zip(seed.locals, binder_names, types, values))
    ):
        if value is not None:
            expression = f"let {name} : {type_} := {value}; {expression}"
        else:
            left, right = _ROLE_BRACKETS[local.role if local.role != ":let" else None]
            expression = f"∀ {left}{name} : {type_}{right}, {expression}"
    return expression


# Text seeds ----------------------------------------------------------------------


@dataclass
class _TextPart:
    text: str
    tokens: list[Token] = field(default_factory=list)
    level_tokens: dict[int, str] = field(default_factory=dict)
    universe_tokens: set[int] = field(default_factory=set)


def _lex_part(text: str) -> _TextPart:
    tokens = [token for token in tokenize(text) if token.kind != "newline"]
    refs, universe_tokens = level_references(tokens)
    return _TextPart(text, tokens, {ref.token: ref.name for ref in refs}, universe_tokens)


def _head(token: Token) -> str:
    # `h✝.symm` refers to `h✝`; a quoted component is never split.
    if token.text.startswith("«"):
        return token.text
    return token.text.split(".", 1)[0]


def _render_text(seed: LeanGoalSeed) -> GoalReplaySpec:
    parts = []
    for local in seed.locals:
        parts.append([_lex_part(local.type)] + ([_lex_part(local.value)] if local.value else []))
    target = _lex_part(seed.target)

    display = {local.name for local in seed.locals}
    references = [
        {_head(token) for part in group for token in part.tokens if token.kind == "ident"} & display
        for group in parts
    ]
    references.append(
        {_head(token) for token in target.tokens if token.kind == "ident"} & display
    )
    binder_names = _plan_names(seed, references, set())

    level_names: list[str] = []
    for part in [*(p for group in parts for p in group), target]:
        for name in part.level_tokens.values():
            if name not in level_names:
                level_names.append(name)
    taken = {token.text for group in parts for p in group for token in p.tokens}
    taken |= {token.text for token in target.tokens}
    levels = level_aliases(level_names, taken)

    def rewrite(part: _TextPart, visible: dict[str, str]) -> str:
        pieces, previous = [], None
        for index, token in enumerate(part.tokens):
            text = token.text
            if index in part.level_tokens:
                text = levels[part.level_tokens[index]]
            elif token.kind == "ident" and index not in part.universe_tokens:
                head = _head(token)
                if head in visible and visible[head] != head:
                    text = visible[head] + token.text[len(head):]
            if previous is not None and token.start > previous.end:
                pieces.append(" ")
            pieces.append(text)
            previous = token
        return "".join(pieces)

    types, values, visible = [], [], {}
    for local, binder, group in zip(seed.locals, binder_names, parts):
        types.append(rewrite(group[0], visible))
        values.append(rewrite(group[1], visible) if len(group) > 1 else None)
        visible[local.name] = binder
    rendered_target = rewrite(target, visible)
    expression = _wrap(seed, binder_names, types, values, rendered_target)
    intro, expected, renames = _intro_and_expected(seed, binder_names)
    level_tuple = tuple(levels[name] for name in level_names)
    payload = {"expression": expression, "intro": intro, "levels": list(level_tuple)}
    return GoalReplaySpec(
        expression=expression,
        local_names=tuple(intro),
        universe_levels=level_tuple,
        expected_names=tuple(expected),
        name_map=tuple(renames),
        level_map=tuple((name, alias) for name, alias in levels.items() if name != alias),
        fingerprint=_fingerprint(seed, payload),
    )


# Structured seeds ------------------------------------------------------------------


def _const_roots(names: set[str]) -> set[str]:
    roots = set()
    for name in names:
        roots.add(name if name.startswith("«") else name.split(".", 1)[0])
    return roots


def _render_level(level: R.Level, aliases: dict[str, str]) -> str:
    if isinstance(level, R.LevelNum):
        return str(level.value)
    if isinstance(level, R.LevelParam):
        return aliases[level.name]
    if isinstance(level, R.LevelOffset):
        return f"({_render_level(level.base, aliases)}+{level.offset})"
    keyword = "imax" if level.imax else "max"
    return f"({keyword} {_render_level(level.left, aliases)} {_render_level(level.right, aliases)})"


def _string_literal(value: str) -> str:
    escaped = []
    for char in value:
        if char in '"\\':
            escaped.append("\\" + char)
        elif char == "\n":
            escaped.append("\\n")
        elif char == "\t":
            escaped.append("\\t")
        elif ord(char) < 0x20:
            escaped.append(f"\\x{ord(char):02x}")
        else:
            escaped.append(char)
    return '"' + "".join(escaped) + '"'


class _RawRenderer:
    """Render raw expressions as fully explicit Lean terms.

    Every constant and local is written with ``@`` because Pantograph prints
    implicit and instance arguments; Lean must not insert them a second time.
    Bound names keep their source spelling unless that would capture another
    name; Pantograph's sanitized ``_`` binders get a fresh name only when used.
    """

    def __init__(
        self,
        locals_: tuple[SeedLocal, ...],
        binder_names: list[str],
        reserved: set[str],
        levels: dict[str, str],
    ):
        self.local_names = {local.internal_id: name for local, name in zip(locals_, binder_names)}
        self.levels = levels
        self.used = set(reserved)
        self.memo: dict[int, frozenset[int]] = {}
        self.outer: dict[int, frozenset[str]] = {}

    def _outer_names(self, expr: R.RawExpr) -> frozenset[str]:
        """Locals and constant roots ``expr`` refers to by name."""
        cached = self.outer.get(id(expr))
        if cached is None:
            if isinstance(expr, R.FVar):
                cached = frozenset({self.local_names[expr.id]})
            elif isinstance(expr, R.Const):
                cached = frozenset(_const_roots({expr.name}))
            else:
                cached = frozenset()
                for child in R.children(expr):
                    cached |= self._outer_names(child)
            self.outer[id(expr)] = cached
        return cached

    def render(self, expr: R.RawExpr, bound: list[str]) -> str:
        if isinstance(expr, R.Const):
            return f"@{expr.name}"
        if isinstance(expr, R.FVar):
            return f"@{self.local_names[expr.id]}"
        if isinstance(expr, R.BVar):
            if expr.index >= len(bound):
                raise ReplayInputError(
                    ReplayFailure.MALFORMED_STRUCTURED, f"Loose bound variable {expr.index}."
                )
            return f"@{bound[expr.index]}"
        if isinstance(expr, R.Sort):
            return f"(Sort {_render_level(expr.level, self.levels)})"
        if isinstance(expr, R.NatLit):
            return f"(nat_lit {expr.value})"
        if isinstance(expr, R.StrLit):
            return _string_literal(expr.value)
        if isinstance(expr, R.Placeholder):
            return "_"
        if isinstance(expr, R.App):
            pieces = [self.render(expr.fn, bound), *(self.render(arg, bound) for arg in expr.args)]
            return f"({' '.join(pieces)})"
        if isinstance(expr, R.Let):
            name = self._binder_name(expr.name, expr.body, bound)
            type_ = self.render(expr.type, bound)
            value = self.render(expr.value, bound)
            body = self.render(expr.body, [name, *bound])
            return f"(let {name} : {type_} := {value}; {body})"
        name = self._binder_name(expr.name, expr.body, bound)
        type_ = self.render(expr.type, bound)
        body = self.render(expr.body, [name, *bound])
        left, right = _INFO_BRACKETS[expr.info]
        if expr.kind == "forall":
            return f"(∀ {left}{name} : {type_}{right}, {body})"
        return f"(fun {left}{name} : {type_}{right} => {body})"

    def _binder_name(self, name: str, body: R.RawExpr, bound: list[str]) -> str:
        loose = R.loose_bvars(body, self.memo)
        if name == "_" and 0 not in loose:
            return "_"
        # Names the body still needs from outside this binder must stay visible.
        needed = {bound[i - 1] for i in loose if 0 < i <= len(bound)}
        needed |= self._outer_names(body)
        if name != "_" and name not in needed and "✝" not in name:
            return name
        return _fresh("x" if name == "_" else display_base(name), self.used)


def _render_structured(seed: LeanGoalSeed) -> GoalReplaySpec:
    roots = set()
    level_names: list[str] = []
    for part in seed._raw_parts():
        roots |= _const_roots(R.constants(part))
        R.universe_params(part, level_names)
    references = [
        R.free_vars(local.type) | (R.free_vars(local.value) if local.value is not None else set())
        for local in seed.locals
    ]
    references.append(R.free_vars(seed.target))
    binder_names = _plan_names(seed, references, roots)
    levels = level_aliases(level_names, roots | set(binder_names))
    source_binders = {
        node.name
        for part in seed._raw_parts()
        for node in R.walk(part)
        if isinstance(node, (R.Binder, R.Let))
    }

    renderer = _RawRenderer(
        seed.locals, binder_names, roots | set(binder_names) | source_binders, levels
    )
    types = [renderer.render(local.type, []) for local in seed.locals]
    values = [
        renderer.render(local.value, []) if local.value is not None else None
        for local in seed.locals
    ]
    target = renderer.render(seed.target, [])
    expression = _wrap(seed, binder_names, types, values, target)
    intro, expected, renames = _intro_and_expected(seed, binder_names)
    level_tuple = tuple(levels[name] for name in level_names)
    payload = {"expression": expression, "intro": intro, "levels": list(level_tuple)}
    return GoalReplaySpec(
        expression=expression,
        local_names=tuple(intro),
        universe_levels=level_tuple,
        expected_names=tuple(expected),
        name_map=tuple(renames),
        level_map=tuple((name, alias) for name, alias in levels.items() if name != alias),
        fingerprint=_fingerprint(seed, payload),
    )
