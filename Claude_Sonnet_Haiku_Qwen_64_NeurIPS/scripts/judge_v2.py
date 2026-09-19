"""
Judge with Qwen 3B using Phase 1's EXACT prompts.
Loads comparison trees, judges every level, saves eval stats.
"""
import json, os, re, sys
from pathlib import Path

# ---------- Phase 1 prompts (verbatim from simple_eval.py) ----------
PROMPT_FAITH = open("phase1_prompts/faithfulness.txt").read() if Path("phase1_prompts/faithfulness.txt").exists() else None
# We'll load prompts from files instead of inline to keep exactness
