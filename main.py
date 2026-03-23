import argparse
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.collector import collect_country
from src.preprocessor import load_tree
from src.ted import compute_ted, save_edit_script
from src.patcher import patch
from src.postprocessor import postprocess, tree_to_xml_string
from models.tree import TreeUtils


def run_pipeline(source: str, target: str, fmt: str, overwrite: bool) -> None:
    print("\n" + "=" * 60)
    print(f"  Wikipedia TED Pipeline")
    print(f"  {source} → {target}")
    print("=" * 60)

    print(f"\n[1/6] Collecting data...")
    _collect(source, overwrite)
    _collect(target, overwrite)

    print(f"\n[2/6] Preprocessing trees...")
    t1 = load_tree(source)
    t2 = load_tree(target)
    print(f"      {source}: {_node_count(t1)} nodes")
    print(f"      {target}: {_node_count(t2)} nodes")

    print(f"\n[3/6] Computing TED and extracting edit script...")
    t1_copy = copy.deepcopy(t1)
    edit_script = compute_ted(t1, t2, source=source, target=target)
    diff_path = save_edit_script(edit_script)
    _print_edit_script_summary(edit_script)
    print(f"      Diff saved to: {diff_path}")

    print(f"\n[4/6] Patching {source} tree → {target}...")
    patched_tree = patch(t1_copy, edit_script)
    print(f"      Patched tree: {_node_count(patched_tree)} nodes")

    print(f"\n[5/6] Validating: patch(T1, diff(T1,T2)) ≈ T2...")
    _validate(patched_tree, t2, source, target)

    print(f"\n[6/6] Post-processing output ({fmt})...")
    output_path = postprocess(patched_tree, f"{source}_patched_to_{target}", fmt=fmt)
    print(f"      Output saved to: {output_path}")

    print("\n" + "=" * 60)
    print(f"  Done. TED score: {edit_script.ted_score}")
    print("=" * 60 + "\n")



def _validate(patched: object, target: object, src_name: str, tgt_name: str) -> None:
    patched_xml = tree_to_xml_string(patched)
    target_xml = tree_to_xml_string(target)

    if patched_xml == target_xml:
        print("      ✓ PASS: patched tree matches target exactly.")
        return

    patched_nodes = TreeUtils.postorder(patched)
    target_nodes = TreeUtils.postorder(target)

    patched_labels = _collect_structural_labels(patched)
    target_labels = _collect_structural_labels(target)

    shared = patched_labels & target_labels
    missing = target_labels - patched_labels
    extra = patched_labels - target_labels

    total = len(target_labels)
    coverage = len(shared) / total * 100 if total else 100

    print(f"      ~ PARTIAL: {len(patched_nodes)} patched vs {len(target_nodes)} target nodes")
    print(f"      ~ Structural coverage: {coverage:.1f}%")
    if missing:
        sample = sorted(missing)[:8]
        print(f"      ~ Missing from target: {sample}{'...' if len(missing) > 8 else ''}")
    if extra:
        sample = sorted(extra)[:8]
        print(f"      ~ Extra in patched: {sample}{'...' if len(extra) > 8 else ''}")


def _collect_structural_labels(root) -> set[str]:
    labels = set()
    def walk(node, prefix=""):
        if not node.is_content:
            path = f"{prefix}/{node.label}" if prefix else node.label
            labels.add(path)
            for child in node.children:
                walk(child, path)
    walk(root)
    return labels


def _collect(country: str, overwrite: bool) -> None:
    try:
        path = collect_country(country, overwrite=overwrite)
        print(f"      [OK] {country} -> {path}")
    except Exception as e:
        print(f"      [FAIL] {country}: {e}")
        sys.exit(1)


def _node_count(root) -> int:
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
