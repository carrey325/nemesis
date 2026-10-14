"""Render the committed research aggregates with a restrained, print-like style."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "evidence/results.json").read_text(encoding="utf-8"))
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
INK, MUTED, ACCENT, LIGHT, GRID = "#2D3238", "#858B92", "#576D7C", "#B8C1C7", "#ECEEF0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#D9DDE0", "axes.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "svg.fonttype": "none", "svg.hashsalt": "nemesis-editorial-v2",
})


def save(fig, name):
    for ext in ("svg", "png"):
        kwargs = {"metadata": {"Date": None}} if ext == "svg" else {}
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.22, **kwargs)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8", newline="\n")
    plt.close(fig)


def prediction():
    fig, ax = plt.subplots(figsize=(8.5, 4))
    fig.subplots_adjust(left=.11, right=.96, bottom=.18, top=.91)
    people = DATA["prediction"]["players"]
    values = [p["nll_gain"] for p in people]
    ax.hlines(range(8), 0, values, color=LIGHT, lw=2)
    ax.scatter(values, range(8), s=35, color=ACCENT, zorder=3)
    ax.set_yticks(range(8), [p["player"] for p in people])
    ax.invert_yaxis()
    ax.set_xlim(0, .145)
    ax.set_xlabel("NLL reduction", labelpad=10)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    mean = DATA["prediction"]["mean_nll_gain"]
    ax.axvline(mean, color=LIGHT, ls="--", lw=1)
    ax.text(mean, 1.04, f"mean {mean:.4f}", transform=ax.get_xaxis_transform(),
            ha="center", color=MUTED, size=9)
    for i, value in enumerate(values):
        ax.text(value + .0025, i, f"{value:.3f}", va="center", size=9, color=ACCENT,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    save(fig, "prediction_gain")


def identity():
    fig, ax = plt.subplots(figsize=(7, 5.4))
    matrix = np.array(DATA["prediction"]["identity_gain_matrix"])
    bound = np.abs(matrix).max()
    cmap = LinearSegmentedColormap.from_list("nemesis", ["#B6A396", "#FAFAF8", ACCENT])
    im = ax.imshow(matrix, cmap=cmap, vmin=-bound, vmax=bound)
    labels = [p["player"] for p in DATA["prediction"]["players"]]
    ax.set_xticks(range(8), labels)
    ax.set_yticks(range(8), labels)
    ax.set_xlabel("Target player", labelpad=12)
    ax.set_ylabel("Personal model", labelpad=12)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(8):
        ax.text(i, i, f"{matrix[i, i]:+.3f}", ha="center", va="center", size=9,
                color="white" if matrix[i, i] > bound * .70 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.05, ticks=[-.1, 0, .1])
    cbar.set_label("NLL reduction", labelpad=12)
    cbar.outline.set_visible(False)
    save(fig, "identity_prediction")


def training():
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.90, hspace=.55, wspace=.35)
    for ax, player in zip(axes.flat, DATA["prediction"]["players"]):
        runs = [r for r in DATA["calibration_histories"] if r["model_id"].split("_")[1] == player["player"]]
        for run, color, style in zip(runs, (ACCENT, "#949CA2", "#C5CBCF"), ("-", "--", ":")):
            x = [e["epoch"] for e in run["epochs"]]
            raw = np.array([e["calibration_nll"] for e in run["epochs"]])
            y = raw - raw[0]
            best = int(np.argmin(raw))
            ax.plot(x, y, color=color, ls=style, lw=1.5, label=run["model_id"].split("_")[-1])
            ax.scatter(x[best], y[best], s=24, facecolors="white", edgecolors=color, zorder=4)
        ax.set_title(player["player"], loc="left", fontsize=10)
        ax.grid(axis="y", color=GRID)
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=8)
    fig.supxlabel("Epoch", y=.105, fontsize=10)
    fig.supylabel("Change in calibration NLL", x=.005, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["Seed " + x for x in labels], loc="lower center", ncol=3, frameon=False, fontsize=9)
    save(fig, "training_curves")


def lora():
    row = DATA["lora_engineering"]["overfit"]
    fig, ax = plt.subplots(figsize=(7.7, 3.5))
    fig.subplots_adjust(left=.12, right=.95, bottom=.19, top=.91)
    values = row["losses"]
    ax.plot(range(1, len(values) + 1), values, color=ACCENT, lw=1.8)
    ax.scatter(1, row["initial_loss"], s=24, color=ACCENT)
    ax.annotate(f'{row["initial_loss"]:.3f}', (1, row["initial_loss"]), xytext=(8, 5),
                textcoords="offset points", fontsize=9)
    ax.scatter(len(values) + 1, row["final_loss"], s=38, facecolors="white", edgecolors=ACCENT,
               label="Final evaluation")
    ax.annotate(f'{row["final_loss"]:.3f}', (len(values) + 1, row["final_loss"]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlabel("Step", labelpad=10)
    ax.set_ylabel("Training loss", labelpad=10)
    ax.grid(axis="y", color=GRID)
    ax.set_xlim(.5, len(values) + 1.7)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    save(fig, "lora_engineering")


def interval(ax, row, key, y, color):
    mean = row["gain"]
    low, high = row[key]
    ax.errorbar(mean, y, xerr=np.array([[mean-low], [high-mean]]),
                fmt="o", color=color, capsize=4, ms=5, lw=1.6)


def decision_axis(ax):
    ax.set_ylim(-.6, 1.6)
    ax.axvline(0, color=LIGHT, lw=1, ls="--")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.3f}"))


def decisions():
    fig, ax = plt.subplots(figsize=(8.2, 2.5))
    fig.subplots_adjust(left=.26, right=.95, bottom=.29, top=.90)
    row = DATA["decision"]["E3"]
    for y, key in ((1, "ci95_fixed_players"), (0, "ci95_player_and_game")):
        interval(ax, row, key, y, ACCENT)
    ax.set_yticks([1, 0], ["Fixed players", "Players + games"])
    ax.set_xlim(-.00045, .0039)
    decision_axis(ax)
    ax.set_xticks([0, .001, .002, .003])
    ax.set_xlabel("Maia-3 · expected-score gain", labelpad=12)
    save(fig, "decision_gain")


def decision_all():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.2))
    fig.subplots_adjust(left=.10, right=.98, bottom=.26, top=.80, wspace=.35)
    for ax, key, title in zip(axes, ("ci95_fixed_players", "ci95_player_and_game"),
                              ("Fixed players", "Players + games")):
        for y, family, color in ((1, "E3", ACCENT), (0, "E2", "#9CA6AD")):
            interval(ax, DATA["decision"][family], key, y, color)
        ax.set_yticks([1, 0], ["Maia-3", "Maia-2"])
        ax.set_xlim(-.0038, .004)
        ax.set_title(title, loc="left", fontsize=11, pad=15)
        decision_axis(ax)
        ax.set_xlabel("Expected-score gain", labelpad=12)
    save(fig, "decision_all")


def conversion():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.22, top=.85, wspace=.30)
    rows = DATA["prediction_to_decision"]
    offsets = {"P006": (-20, 9), "P007": (-30, 9), "P013": (5, 8), "P014": (7, 9),
               "P025": (-35, -17), "P029": (6, -18), "P030": (6, -17), "P032": (5, 9)}
    for ax, family in zip(axes, ("E3", "E2")):
        ax.scatter([r["recorded_NLL_gain"] for r in rows], [r[family+"_G"] for r in rows],
                   s=30, color=ACCENT, zorder=3)
        for row in rows:
            offset = offsets[row["player_id"]]
            if family == "E2" and row["player_id"] == "P029":
                offset = (-35, 8)
            elif family == "E2" and row["player_id"] == "P014":
                offset = (7, -14)
            ax.annotate(row["player_id"], (row["recorded_NLL_gain"], row[family+"_G"]),
                        xytext=offset, textcoords="offset points", fontsize=8, color=MUTED)
        ax.axhline(0, color=LIGHT, ls="--", lw=1)
        ax.set_xlim(0, .145)
        ax.set_ylim(-.0025, .0042)
        ax.set_title("Maia-3" if family == "E3" else "Maia-2", loc="left", fontsize=11)
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("NLL reduction", labelpad=10)
    axes[0].set_ylabel("Expected-score gain", labelpad=10)
    save(fig, "prediction_to_decision")


if __name__ == "__main__":
    for function in (prediction, identity, training, lora, decisions, decision_all, conversion):
        function()
    print("Rendered seven figures in SVG and PNG from evidence/results.json.")
