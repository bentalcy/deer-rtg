import csv
from pathlib import Path


def load_clusters(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    overrides: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = row.get("image_path")
            new_cluster = row.get("new_cluster_id")
            if image_path and new_cluster:
                overrides[image_path] = new_cluster
    return overrides


def load_merges(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    merges: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            src = row.get("from_cluster")
            dst = row.get("to_cluster")
            if src and dst:
                merges[src] = dst
    return merges


def resolve_merge(cluster_id: str, merges: dict[str, str]) -> str:
    seen = set()
    current = cluster_id
    while current in merges and current not in seen:
        seen.add(current)
        current = merges[current]
    return current


def apply_edits(
    base_rows: list[dict[str, str]],
    overrides: dict[str, str],
    merges: dict[str, str],
) -> list[dict[str, str]]:
    updated = []
    for row in base_rows:
        image_path = row.get("image_path", "")
        cluster_id = row.get("cluster_id", "")

        if image_path in overrides:
            cluster_id = overrides[image_path]

        if cluster_id:
            cluster_id = resolve_merge(cluster_id, merges)

        new_row = dict(row)
        new_row["cluster_id"] = cluster_id
        updated.append(new_row)
    return updated


def write_clusters(out_csv: Path, rows: list[dict[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default="data/reid/clusters_hdbscan.csv")
    parser.add_argument("--overrides", default="data/reid/cluster_overrides.csv")
    parser.add_argument("--merges", default="data/reid/cluster_merges.csv")
    parser.add_argument("--out", default="data/reid/clusters_corrected.csv")
    args = parser.parse_args()

    base_rows = load_clusters(Path(args.clusters))
    overrides = load_overrides(Path(args.overrides))
    merges = load_merges(Path(args.merges))
    updated = apply_edits(base_rows, overrides, merges)
    write_clusters(Path(args.out), updated)
    print("Wrote corrected clusters to", args.out)


if __name__ == "__main__":
    main()
