"""
classify_direction.py  —  K-means direction clustering with auto-k via silhouette score.

Usage:
    python classify_direction.py out/<stem>_position.csv

Output:
    out/<stem>_direction.csv   (car_id, direction_cluster, angle_deg)

Requires:
    pip install scikit-learn
"""

import os
import sys
import csv
import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# --- Config
MIN_DURATION_S    = 2.0   # skip cars tracked for less than this
MIN_DISPLACEMENT_M = 2.0  # skip parked / barely moved cars

K_MIN, K_MAX = 2, 6      # search range for number of direction clusters

# -------------------------------------------------------------------

def load_positions(csv_path):
    tracks = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["car_id"])
            tracks[cid].append((float(row["time_s"]), float(row["x_m"]), float(row["z_m"])))
    return tracks


def compute_displacement_vectors(tracks):
    """Return (car_ids, X) where X is an (N,2) array of unit direction vectors."""
    car_ids, vectors = [], []
    skipped = 0
    for cid, pts in tracks.items():
        pts.sort(key=lambda p: p[0])
        t0, x0, z0 = pts[0]
        t1, x1, z1 = pts[-1]

        dur  = t1 - t0
        dx   = x1 - x0
        dz   = z1 - z0
        disp = math.hypot(dx, dz)

        if dur < MIN_DURATION_S or disp < MIN_DISPLACEMENT_M:
            skipped += 1
            continue

        car_ids.append(cid)
        vectors.append([dx, dz])

    print(f"  {len(car_ids)} valid tracks  ({skipped} skipped — too short or stationary)")
    return car_ids, np.array(vectors, dtype=np.float32)


def pick_best_k(X_unit):
    """Try k = K_MIN..K_MAX, return k with highest silhouette score."""
    print(f"\n  Silhouette scores (k={K_MIN}..{K_MAX}):")
    best_k, best_score, best_labels = K_MIN, -1.0, None

    for k in range(K_MIN, min(K_MAX + 1, len(X_unit))):
        km     = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_unit)
        score  = silhouette_score(X_unit, labels)
        marker = " <--" if score > best_score else ""
        print(f"    k={k}  silhouette={score:.4f}{marker}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    print(f"\n  Best k={best_k}  silhouette={best_score:.4f}")
    return best_k, best_labels


def cluster_summary(car_ids, vectors, labels, best_k):
    print(f"\n  Cluster summary:")
    for cl in range(best_k):
        idxs   = [i for i, l in enumerate(labels) if l == cl]
        angles = [math.degrees(math.atan2(vectors[i][0], vectors[i][1])) for i in idxs]
        mean_a = float(np.mean(angles))
        print(f"    Cluster {cl}: {len(idxs):3d} cars   mean angle = {mean_a:+.1f}°")


def write_output(out_path, car_ids, vectors, labels):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["car_id", "direction_cluster", "angle_deg"])
        for i, cid in enumerate(car_ids):
            angle = math.degrees(math.atan2(vectors[i][0], vectors[i][1]))
            w.writerow([cid, int(labels[i]), round(angle, 1)])
    print(f"\n  Wrote {out_path} ({len(car_ids)} rows)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_direction.py out/<stem>_position.csv")
        sys.exit(1)

    pos_path = sys.argv[1]
    if not os.path.exists(pos_path):
        print(f"File not found: {pos_path}")
        sys.exit(1)

    print(f"Loading {pos_path} ...")
    tracks = load_positions(pos_path)
    print(f"  {len(tracks)} unique car IDs found")

    car_ids, vectors = compute_displacement_vectors(tracks)
    if len(car_ids) < K_MIN:
        print(f"Not enough valid tracks ({len(car_ids)}) to cluster — need at least {K_MIN}.")
        sys.exit(1)

    X_unit = normalize(vectors)

    best_k, labels = pick_best_k(X_unit)
    cluster_summary(car_ids, vectors, labels, best_k)

    stem     = pos_path.replace("_position.csv", "")
    out_path = f"{stem}_direction.csv"
    write_output(out_path, car_ids, vectors, labels)


if __name__ == "__main__":
    main()
