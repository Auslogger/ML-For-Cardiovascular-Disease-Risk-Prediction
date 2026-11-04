"""
Build the handful of figures shown in the README from a finished pipeline run.

    python main.py                         # trains + evaluates everything (~20 min)
    python scripts/make_readme_figures.py  # writes docs/figures/*.png

The pipeline writes dozens of timestamped plots to reports/ (git-ignored). This
script picks the latest Phase 7/8 outputs, re-creates the same held-out test split,
and draws a small, consistent set of figures that are committed to the repo.
It also prints the results table used in the README.
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FIGURES_DIR, PHASE8_PRIMARY_MODEL, PROCESSED_DATA_DIR, TABLES_DIR,
)
from src.data_preprocessing import make_train_val_test_split  # noqa: E402
from src.evaluation import Phase7Config, _net_benefit, load_models_for_phase7  # noqa: E402

OUT_DIR = ROOT / "docs" / "figures"

# Light chart surface + the first three slots of a colour-blind-safe categorical
# palette. Colours follow the entity: best model = blue, baseline = orange.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#c9c8c2",
)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

NAMES = {
    "baseline_logistic": "Logistic regression (baseline)",
    "svm": "SVM", "random_forest": "Random forest", "extra_trees": "Extra trees",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "ebm": "EBM",
}


def pretty(key: str, with_suffix: bool = False) -> str:
    if key == "baseline_logistic":
        return NAMES[key]
    base = key.replace("_tuned_calibrated", "").replace("_tuned", "")
    name = NAMES.get(base, base)
    if with_suffix:
        name += " (calibrated)" if key.endswith("_calibrated") else " (uncalibrated)"
    return name


def latest(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No files matching {pattern} in {directory} - run `python main.py` first.")
    return files[-1]


def style_axes(ax, title: str, xlabel: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10.5)


def new_figure(size=(8, 5)):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    return fig, ax


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def legend(ax, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=10, labelcolor=INK_2, **kwargs)
    for handle in leg.legend_handles:
        handle.set_alpha(1)


# ------------------------------------------------------------------------- figures

def plot_model_comparison(table: pd.DataFrame, best: str) -> None:
    """Test-set ROC-AUC with 95% bootstrap CIs; best tuned model and baseline highlighted."""
    df = table.sort_values("roc_auc").reset_index(drop=True)
    baseline_auc = df.loc[df["model"] == "baseline_logistic", "roc_auc"].iloc[0]
    title = ("No tuned model beats plain logistic regression"
             if baseline_auc >= df.loc[df["model"] == best, "roc_auc"].iloc[0]
             else "Every model lands in the same band")
    fig, ax = new_figure((8, 4.8))

    for i, row in df.iterrows():
        colour = BLUE if row["model"] == best else ORANGE if row["model"] == "baseline_logistic" else MUTED
        ax.plot([row["auc_lower"], row["auc_upper"]], [i, i], color=colour, linewidth=2,
                solid_capstyle="round", alpha=0.55 if colour == MUTED else 0.8)
        ax.scatter(row["roc_auc"], i, s=70, color=colour, zorder=3, edgecolors=SURFACE, linewidths=2)
        if row["model"] in (best, "baseline_logistic"):
            ax.annotate(f"{row['roc_auc']:.3f}", (row["auc_upper"], i), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=10)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([pretty(m) for m in df["model"]])
    lo = np.floor((df["auc_lower"].min() - 0.01) * 50) / 50
    hi = np.ceil((df["auc_upper"].max() + 0.03) * 50) / 50
    ax.set_xlim(lo, hi)
    ax.grid(axis="y", visible=False)
    style_axes(ax, title, "ROC-AUC on the held-out test set (n = 678), with 95% bootstrap interval")
    save(fig, "model_comparison.png")


def plot_roc(y_test, proba: dict[str, np.ndarray], colours: dict[str, str]) -> None:
    fig, ax = new_figure((6, 5.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for key, p in proba.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=colours[key], linewidth=2, label=pretty(key))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    style_axes(ax, "ROC curves (test set)", "False positive rate", "True positive rate")
    legend(ax, loc="lower right")
    save(fig, "roc_curves.png")


def plot_calibration(y_test, uncalibrated: np.ndarray, calibrated: np.ndarray, base_key: str) -> None:
    fig, ax = new_figure((6, 5.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="Perfect calibration")
    upper = 0.0
    for p, colour, label in (
        (uncalibrated, MUTED, "Before calibration"),
        (calibrated, BLUE, "After sigmoid calibration"),
    ):
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        upper = max(upper, frac_pos.max(), mean_pred.max())
        ax.plot(mean_pred, frac_pos, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    lim = min(1.0, np.ceil((upper + 0.05) * 10) / 10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    style_axes(ax, f"Calibration – {pretty(base_key)}", "Predicted 10-year CHD risk",
               "Observed CHD rate (deciles of risk)")
    legend(ax, loc="upper left")
    save(fig, "calibration.png")


def plot_decision_curve(y_test, p_best: np.ndarray, best: str) -> None:
    thresholds = np.linspace(0.05, 0.40, 71)
    prevalence = float(np.mean(y_test))
    model_nb = [_net_benefit(y_test, p_best, t) for t in thresholds]
    treat_all = [prevalence - (1 - prevalence) * t / (1 - t) for t in thresholds]

    fig, ax = new_figure((8, 4.8))
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.plot(thresholds, treat_all, color=MUTED, linewidth=2, label="Treat everyone")
    ax.plot(thresholds, model_nb, color=BLUE, linewidth=2, label=pretty(best, with_suffix=True))
    # Label the zero line in the empty band just under it (treat-all has already left the plot)
    ax.text(0.20, -0.003, "Treat no one", color=INK_2, fontsize=10, ha="left", va="top")
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(min(-0.02, min(model_nb) - 0.005), max(max(model_nb), max(treat_all)) + 0.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_axes(ax, "Decision curve: is acting on the model better than a blanket policy?",
               "Risk threshold at which a clinician would intervene", "Net benefit")
    legend(ax, loc="upper right")
    save(fig, "decision_curve.png")


def copy_shap_summary() -> None:
    src = latest(FIGURES_DIR / "interpretation", f"phase8_shap_beeswarm_top20_{PHASE8_PRIMARY_MODEL}_*.png")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT_DIR / "shap_summary.png")
    print(f"  copied {src.name} -> docs/figures/shap_summary.png")


# ---------------------------------------------------------------------------- main

def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })

    metrics = pd.read_csv(latest(TABLES_DIR, "phase7_model_comparison_*.csv"))
    cis = pd.read_csv(latest(TABLES_DIR, "phase7_bootstrap_cis_*.csv"))
    auc_ci = (cis[cis["metric"] == "roc_auc"].set_index("model")[["lower", "upper"]]
              .rename(columns={"lower": "auc_lower", "upper": "auc_upper"}))
    metrics = metrics.join(auc_ci, on="model")

    # One row per algorithm: the calibrated tuned model, plus the baseline
    headline = metrics[(metrics["model"] == "baseline_logistic")
                       | metrics["model"].str.endswith("_tuned_calibrated")].copy()
    tuned = headline[headline["model"] != "baseline_logistic"].sort_values("roc_auc", ascending=False)
    best, runner_up = tuned.iloc[0]["model"], tuned.iloc[1]["model"]  # best *tuned* models

    train_engineered = pd.read_csv(PROCESSED_DATA_DIR / "train_engineered.csv")
    *_, X_test, _, _, y_test = make_train_val_test_split(train_engineered)

    primary_uncal = PHASE8_PRIMARY_MODEL.replace("_calibrated", "")
    wanted = {best, runner_up, "baseline_logistic", PHASE8_PRIMARY_MODEL, primary_uncal}
    models, *_ = load_models_for_phase7(config=Phase7Config())
    proba = {k: models[k].predict_proba(X_test)[:, 1] for k in wanted}

    print("Writing README figures:")
    plot_model_comparison(headline, best)
    plot_roc(y_test, {k: proba[k] for k in (best, runner_up, "baseline_logistic")},
             {best: BLUE, "baseline_logistic": ORANGE, runner_up: AQUA})
    plot_calibration(y_test, proba[primary_uncal], proba[PHASE8_PRIMARY_MODEL], primary_uncal)
    plot_decision_curve(y_test, proba[best], best)
    copy_shap_summary()

    print("\nResults table (test set, n = %d, %d with CHD):\n" % (len(y_test), int(y_test.sum())))
    print("| Model | ROC-AUC (95% CI) | Avg. precision | Brier | Sensitivity | Specificity |")
    print("|---|---|---|---|---|---|")
    for _, r in headline.sort_values("roc_auc", ascending=False).iterrows():
        print(f"| {pretty(r['model'])} | {r['roc_auc']:.3f} ({r['auc_lower']:.3f}-{r['auc_upper']:.3f}) "
              f"| {r['average_precision']:.3f} | {r['brier_score']:.3f} "
              f"| {r['recall_sensitivity']:.2f} | {r['specificity']:.2f} |")
    print(f"\nPrevalence in test set: {y_test.mean():.3f}")


if __name__ == "__main__":
    main()
