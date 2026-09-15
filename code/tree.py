# tree.py
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from typing import Optional, Union

@dataclass
class Stat: # need to check
    specificity_average: Optional[int] = None
    specificity_range: list[int] = field(default_factory=list)
    entail_1_average: Optional[int] = None
    entail_2_average: Optional[int] = None
    entail_1_range: list[int] = field(default_factory=list)
    entail_2_range: list[int] = field(default_factory=list)
    # --- faithfulness / informativeness split (additive; keep at the end so the
    # positional Stat(...) construction in promote_to_next_level stays valid) ---
    faithfulness_average: Optional[float] = None
    faithfulness_range: list[float] = field(default_factory=list)
    informativeness_average: Optional[float] = None
    informativeness_range: list[float] = field(default_factory=list)
    specificity_faithfulness_average: Optional[float] = None
    specificity_faithfulness_range: list[float] = field(default_factory=list)
    specificity_informativeness_average: Optional[float] = None
    specificity_informativeness_range: list[float] = field(default_factory=list)
# ─────────────────────────────────────────────────────────────
@dataclass
class Node:
    id: int

    parent: list[int] = field(default_factory=list)        # ← list now
    children: list[int] = field(default_factory=list)      # ← unchanged, still list

    prompt: Optional[str] = None

    match: Optional[int] = None
    score: Optional[float] = None
    score2: Optional[float] = None

    mainIdea: Optional[str] = None
    summary: Optional[str] = None
    summary_blocked: Optional[bool] = False
    text_similarity: Optional[float] = None
    idea_similarity: Optional[float] = None
    unmatched: Optional[bool] = False

    evaluation: Optional[str] = None
    evalScore: Optional[int] = None
    entail_1: Optional[int] = None
    entail_2: Optional[int] = None

    alive: bool = True

    third: Optional[int] = None
    third_score: Optional[int] = None
    evaluation_specificity: Optional[str] = None
    specificity: Optional[int] = None

    # --- faithfulness / informativeness split (additive) ---
    # entail context = node vs match ; specificity context = node vs third
    # 1 = average, 2 = minimum (mirrors entail_1 / entail_2)
    faithfulness1: Optional[float] = None
    faithfulness2: Optional[float] = None
    informativeness1: Optional[float] = None
    informativeness2: Optional[float] = None
    evaluation_faithfulness: Optional[str] = None
    evaluation_informativeness: Optional[str] = None
    specificity_faithfulness: Optional[float] = None
    specificity_informativeness: Optional[float] = None
    evaluation_specificity_faithfulness: Optional[str] = None
    evaluation_specificity_informativeness: Optional[str] = None

    wb_score_mistral: Optional[int] = None
    wb_score_llama: Optional[int] = None
    wb_scores: dict = field(default_factory=dict)

    # --- evaltree (additive) ---
    capability: Optional[str] = None      # leaf gerund-phrase annotation
    cluster_size: Optional[int] = None    # number of descendant leaves
    level_num: Optional[int] = None
# ─────────────────────────────────────────────────────────────
@dataclass
class Level:
    level: int
    stat: Stat = field(default_factory=Stat) # need to check
    nodes: list[Node] = field(default_factory=list)
    
# ─────────────────────────────────────────────────────────────
@dataclass
class Tree:
    levels: list[Level]
    next_id: int = 0
    # provenance header: which method/providers built this tree
    # (e.g. {"method": "blossom", "blossom": "claude", ...})
    config: dict = field(default_factory=dict)

    def new_id(self) -> int:
        _id = self.next_id
        self.next_id += 1
        return _id
# ─────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tree":
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise ValueError("Tree.load expects a *.json file")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            raw = payload["levels"]

        lvls = [
            Level(
                level=lvl["level"],
                stat=Stat(**lvl.get("stat", {})),
                nodes=[
                    Node(
                        id=n.get("id", n.get("row")),

                        parent=n.get("parent", []),               # list
                        children=n.get("children", []),           # list

                        prompt=n.get("prompt"),
                        match=n.get("match"),

                        mainIdea=n.get("mainIdea"),
                        summary=n.get("summary"),
                        summary_blocked=n.get("summary_blocked"),
                        text_similarity=n.get("text_similarity"),
                        idea_similarity=n.get("idea_similarity"),
                        score=n.get("score"),
                        score2=n.get("score2"),
                        unmatched=n.get("unmatched"),

                        evaluation=n.get("evaluation"),
                        evalScore = n.get("evalScore"),
                        entail_1 =n.get("entail_1"),
                        entail_2=n.get("entail_2"),

                        alive=n.get("alive", True),

                        third=n.get("third"),
                        third_score = n.get("third_score"),
                        specificity=n.get("specificity"),
                        evaluation_specificity = n.get("evaluation_specificity"),
                        faithfulness1 = n.get("faithfulness1"),
                        faithfulness2 = n.get("faithfulness2"),
                        informativeness1 = n.get("informativeness1"),
                        informativeness2 = n.get("informativeness2"),
                        evaluation_faithfulness = n.get("evaluation_faithfulness"),
                        evaluation_informativeness = n.get("evaluation_informativeness"),
                        specificity_faithfulness = n.get("specificity_faithfulness"),
                        specificity_informativeness = n.get("specificity_informativeness"),
                        evaluation_specificity_faithfulness = n.get("evaluation_specificity_faithfulness"),
                        evaluation_specificity_informativeness = n.get("evaluation_specificity_informativeness"),
                        wb_score_mistral = n.get("wb_score_mistral"),
                        wb_score_llama = n.get("wb_score_llama"),
                        wb_scores = n.get("wb_scores", {}),
                        capability = n.get("capability"),
                        cluster_size = n.get("cluster_size"),
                        level_num = n.get("level_num")
                    )
                    for n in lvl["nodes"]
                ],

            )
            for lvl in raw
        ]

        max_id = max((n.id for lvl in lvls for n in lvl.nodes), default=-1)
        return cls(lvls, next_id=max_id + 1, config=payload.get("config", {}))
# ─────────────────────────────────────────────────────────────
    def dump(self, path: str):
        payload = {}
        if self.config:
            payload["config"] = self.config
        payload.update({
            "levels": [
                {
                    "level": lvl.level,
                    "stat": asdict(lvl.stat),
                    "nodes": [asdict(n) for n in lvl.nodes],
                }
                for lvl in self.levels
            ]
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=Tree._np_encoder)
# ─────────────────────────────────────────────────────────────
    @staticmethod
    def _np_encoder(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"{repr(obj)} is not JSON serialisable")