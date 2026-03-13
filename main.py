import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.collector import collect_country
from src.preprocessor import load_tree
from src.ted import compute_ted, save_edit_script
from src.patcher import patch
from src.postprocessor import postprocess


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(source: str, target: str, fmt: str, overwrite: bool) -> None:
    print("\n" + "=" * 60)
    print(f"  Wikipedia TED Pipeline")
    print(f"  {source} → {target}")
    print("=" * 60)

    print(f"\n[1/5] Collecting data...")
    _collect(source, overwrite)
    _collect(target, overwrite)

    print(f"\n[2/5] Preprocessing trees...")
    t1 = load_tree(source)
    t2 = load_tree(target)
    print(f"      {source}: {_node_count(t1)} nodes")
    print(f"      {target}: {_node_count(t2)} nodes")

    print(f"\n[3/5] Computing TED and extracting edit script...")
    edit_script = compute_ted(t1, t2, source=source, target=target)
    diff_path = save_edit_script(edit_script)
    _print_edit_script_summary(edit_script)
    print(f"      Diff saved to: {diff_path}")

    print(f"\n[4/5] Patching {source} tree → {target}...")
    patched_tree = patch(t1, edit_script)
    print(f"      Patched tree: {_node_count(patched_tree)} nodes")

    print(f"\n[5/5] Post-processing output ({fmt})...")
    output_path = postprocess(patched_tree, f"{source}_patched_to_{target}", fmt=fmt)
    print(f"      Output saved to: {output_path}")

    print("\n" + "=" * 60)
    print(f"  Done. TED score: {edit_script.ted_score}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect(country: str, overwrite: bool) -> None:
    try:
        path = collect_country(country, overwrite=overwrite)
        print(f"      [OK] {country} -> {path}")
    except Exception as e:
        print(f"      [FAIL] {country}: {e}")
        sys.exit(1)


def _node_count(root) -> int:
    from models.tree import TreeUtils
    return len(TreeUtils.postorder(root))


def _print_edit_script_summary(edit_script) -> None:
    inserts = sum(1 for op in edit_script.operations if op.operation == "INSERT")
    deletes = sum(1 for op in edit_script.operations if op.operation == "DELETE")
    renames = sum(1 for op in edit_script.operations if op.operation == "RENAME")

    print(f"      TED score : {edit_script.ted_score}")
    print(f"      Operations: {len(edit_script.operations)} total")
    print(f"        INSERT  : {inserts}")
    print(f"        DELETE  : {deletes}")
    print(f"        RENAME  : {renames}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare and difference two Wikipedia country infoboxes using Tree Edit Distance."
    )
    parser.add_argument("source", type=str, help="Source country name (e.g. 'Lebanon')")
    parser.add_argument("target", type=str, help="Target country name (e.g. 'Switzerland')")
    parser.add_argument(
        "--format",
        type=str,
        choices=["xml", "infobox"],
        default="xml",
        dest="fmt",
        help="Output format for the patched tree (default: xml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-scrape country data even if XML files already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        source=args.source,
        target=args.target,
        fmt=args.fmt,
        overwrite=args.overwrite,
    )