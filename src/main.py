# ─────────────────────────────────────────────────────────────────────────────
# dimensionality-reduction-comparison
# Comparison of dimensionality reduction techniques on MNIST:
#   · Exercise 1 — Training efficiency (reduction time, train time, accuracy)
#   · Exercise 2 — 2D visualization (scatter plots per technique)
# ─────────────────────────────────────────────────────────────────────────────

import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE


# UTILS

def plot_exercise1(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exercise 1 — Training Efficiency Comparison", fontsize=13, fontweight="bold")

    results_df[["reduction_time_s", "train_time_s"]].plot(
        kind="bar", ax=axes[0], colormap="viridis", edgecolor="white"
    )
    axes[0].set_title("Reduction & Training Times")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].tick_params(axis="x", rotation=35)

    results_df["accuracy"].plot(
        kind="bar", ax=axes[1], color="#2ecc71", edgecolor="white"
    )
    axes[1].set_title("Accuracy per Technique")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.show()


def plot_exercise2(projections, y_sub):
    palette = sns.color_palette("tab10", 10)
    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6))
    fig.suptitle("Exercise 2 — 2D Projections  (10k samples)", fontsize=13, fontweight="bold")

    for ax, (name, (X_2d, elapsed)) in zip(axes, projections.items()):
        for digit in range(10):
            mask = y_sub == digit
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       s=2, alpha=0.5, color=palette[digit], label=str(digit))
        ax.set_title(f"{name}\n({elapsed:.1f}s)", fontsize=11)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(markerscale=4, title="Digit", fontsize=7, loc="best")

    plt.tight_layout()
    plt.show()


def print_summary(results_ex1, baseline_acc, baseline_train_time, projections_ex2):
    print("\n" + "═" * 70)
    print("  FINAL SUMMARY REPORT")
    print("═" * 70)

    print("\n── Exercise 1: Training Efficiency ──\n")

    baseline_row = pd.DataFrame([{
        "reduction_time_s": 0.0,
        "train_time_s": round(baseline_train_time, 3),
        "accuracy": round(baseline_acc, 4),
        "n_components": 784,
        "data_used": "full dataset",
    }], index=["Baseline (no reduction)"])

    summary = pd.concat([baseline_row, results_ex1])
    summary["total_time_s"] = summary["reduction_time_s"] + summary["train_time_s"]
    summary["acc_vs_baseline"] = (summary["accuracy"] - baseline_acc).round(4)
    summary.index.name = "Technique"

    print(summary.to_string())

    best_acc = summary["accuracy"].idxmax()
    fastest = summary["total_time_s"].idxmin()
    best_tradeoff = (summary["accuracy"] / summary["total_time_s"]).idxmax()

    print(f"\n  · Best accuracy → {best_acc}  ({summary.loc[best_acc, 'accuracy']:.4f})")
    print(f"  · Fastest total time → {fastest}  ({summary.loc[fastest, 'total_time_s']:.2f}s)")
    print(f"  · Best acc/time → {best_tradeoff}")
    print("\n  Kernel PCA and LLE used a subset (10k train / 2k test).")
    print("Their accuracy is NOT directly comparable to full-dataset results.")

    print("\n── Exercise 2: 2D Projection Timing ──\n")

    df2 = pd.DataFrame(
        [{"technique": name, "projection_time_s": round(elapsed, 2)}
         for name, (_, elapsed) in projections_ex2.items()]
    ).set_index("technique")

    print(df2.to_string())
    fastest_2d = df2["projection_time_s"].idxmin()
    print(f"\n  · Fastest 2D projection → {fastest_2d}  ({df2.loc[fastest_2d, 'projection_time_s']:.2f}s)")
    print("\n  All techniques used 10,000 samples for fair visual comparison.")
    print("t-SNE has no transform() — fitted directly on the subset.")
    print("═" * 70)


# DATA LOADING & INSPECTION

print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=True)
X, y = mnist["data"], mnist["target"]
print("Dataset loaded successfully!")
print(f"Shape of X : {X.shape}")
print(f"Shape of y : {y.shape}")
print(f"Unique labels : {sorted(set(y))}")

# Sample digit
sample = X.iloc[0].values.reshape(28, 28)
plt.figure(figsize=(3, 3))
plt.imshow(sample, cmap="binary")
plt.axis("off")
plt.title(f"Label: {y.iloc[0]}")
plt.tight_layout()
plt.show()

# Class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title("Class Distribution — MNIST")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# TRAIN / TEST SPLIT & PREPROCESSING

X_train, X_test = X.iloc[:60000], X.iloc[60000:]
y_train, y_test = y.iloc[:60000], y.iloc[60000:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# EXERCISE 1 — Training efficiency comparison
#
# For each technique we measure:
#   · Reduction time — how long it takes to fit and transform the data
#   · Training time - how long Logistic Regression takes on the reduced data
#   · Accuracy — classification accuracy on the test set
#
# NOTE: Kernel PCA and LLE are evaluated on a subset (10k train / 2k test)
# due to their high computational complexity (O(n²)–O(n³)).
# Their results are not directly comparable to full-dataset techniques.

print("\n" + "═" * 60)
print("  EXERCISE 1 — Training Efficiency")
print("═" * 60)

# Baseline — Logistic Regression on full scaled data, no reduction
baseline_model = LogisticRegression(max_iter=1000, n_jobs=-1)
t0 = time.perf_counter()
baseline_model.fit(X_train_scaled, y_train)
baseline_train_time = time.perf_counter() - t0
baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test_scaled))
print(f"\nBaseline  |  train: {baseline_train_time:.2f}s  |  accuracy: {baseline_acc:.4f}")

techniques_ex1 = {
    "PCA": (PCA(n_components=50), False),
    "Randomized PCA": (PCA(n_components=50, svd_solver="randomized"), False),
    "Incremental PCA": (IncrementalPCA(n_components=50), False),
    "Kernel PCA": (KernelPCA(n_components=30, kernel="rbf"), True),
    "LLE": (LocallyLinearEmbedding(n_components=30), True),
}

results_ex1 = {}

for name, (reducer, use_subset) in techniques_ex1.items():

    # Kernel PCA and LLE use a subset due to high computational cost
    if use_subset:
        X_tr, y_tr = X_train_scaled[:10000], y_train.iloc[:10000]
        X_te, y_te = X_test_scaled[:2000],   y_test.iloc[:2000]
        subset_note = "subset (10k/2k)"
    else:
        X_tr, y_tr = X_train_scaled, y_train
        X_te, y_te = X_test_scaled,  y_test
        subset_note = "full dataset"

    t0 = time.perf_counter()
    X_tr_red = reducer.fit_transform(X_tr)
    if not hasattr(reducer, "transform"):
        print(f"{name}: no transform() available — skipped")
        continue
    X_te_red = reducer.transform(X_te)
    reduction_time = time.perf_counter() - t0

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X_tr_red, y_tr)
    train_time = time.perf_counter() - t0

    acc = accuracy_score(y_te, model.predict(X_te_red))
    n_comp = reducer.n_components if hasattr(reducer, "n_components") else "—"

    results_ex1[name] = {
        "reduction_time_s": round(reduction_time, 3),
        "train_time_s": round(train_time, 3),
        "accuracy": round(acc, 4),
        "n_components": n_comp,
        "data_used": subset_note,
    }
    print(f"{name:<18} | reduce: {reduction_time:.2f}s | train: {train_time:.2f}s "
          f"| acc: {acc:.4f} | {subset_note}")

results_ex1_df = pd.DataFrame(results_ex1).T
plot_exercise1(results_ex1_df)

# EXERCISE 2 — 2D Visualization
#
# Goal: compare how each technique projects MNIST into 2D and whether
#       the digit clusters are visually separable.
#
# All techniques use the same subset of 10,000 samples for fair comparison
# and to keep t-SNE, Kernel PCA and LLE computationally feasible.
# t-SNE does not support transform(), so it is fitted directly on the subset.

print("\n" + "═" * 60)
print("  EXERCISE 2 — 2D Visualization")
print("═" * 60)

N = 10000
X_sub = X_train_scaled[:N]
y_sub = y_train.iloc[:N].astype(int).values

techniques_ex2 = {
    "PCA": PCA(n_components=2),
    "Kernel PCA": KernelPCA(n_components=2, kernel="rbf", gamma=0.01),
    "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=15),
    "t-SNE": TSNE(n_components=2, perplexity=40,
                       random_state=42, n_jobs=-1),
}

projections_ex2 = {}

for name, reducer in techniques_ex2.items():
    print(f"  Projecting with {name}...", end=" ", flush=True)
    t0 = time.perf_counter()
    X_2d = reducer.fit_transform(X_sub)
    elapsed = time.perf_counter() - t0
    projections_ex2[name] = (X_2d, elapsed)
    print(f"done ({elapsed:.1f}s)")

plot_exercise2(projections_ex2, y_sub)

# SUMMARY REPORT

print_summary(results_ex1_df, baseline_acc, baseline_train_time, projections_ex2)