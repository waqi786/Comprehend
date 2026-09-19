"""
Add `third` field to SciNCL trees by running find_third with SciNCL model.
"""
import sys
from pathlib import Path

# Add paths so we can import tree and third_neighbor
COMP = Path("/groups/hawshiuan/waqar/COMPREHEND_public")
LOCAL_CODE = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(LOCAL_CODE))
sys.path.insert(0, str(COMP))

from tree import Tree
from retrieval.third_neighbor import find_third

for subset in ["similar", "dissimilar"]:
    f = f"results/UF_random_prompts_64_base_{subset}_haiku_scincl_blossom.json"
    print(f"\n=== {subset.upper()} ===")
    tree = Tree.load(f)
    for li in range(6):  # L0..L5
        print(f"\n--- find_third L{li} ---")
        find_third(tree, li, model_name="malteos/scincl")
    tree.dump(f)
    print(f"Re-saved {f}")
