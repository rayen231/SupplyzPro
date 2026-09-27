"""Render README charts from the stored evaluation results.

Run: python -m scripts.make_charts   ->  docs/images/*.png
"""
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from app import config  # noqa: E402
from app.ingestion.loader import load_ground_truth  # noqa: E402
from app.models.finding import FAILURE_CODES, FAILURE_LABELS  # noqa: E402
from app.services import store  # noqa: E402

OUT = config.ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CONFIGS = [("rules", "Rules only"), ("llm:openai/gpt-oss-20b", "LLM only · 20B"),
           ("hybrid:openai/gpt-oss-20b", "Hybrid · 20B"), ("llm:openai/gpt-oss-120b", "LLM only · 120B"),
           ("hybrid:openai/gpt-oss-120b", "Hybrid · 120B")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig, title, subtitle):
    fig.text(0.02, 0.965, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=10.5, color=INK2, va="top")


metrics = {c["config"]: c["metrics"] for c in store.list_configs()}
gt = load_ground_truth()

# ---------------------------------------------------------------- 1. F1 all vs held-out
fig, ax = plt.subplots(figsize=(9, 4.6))
fig.subplots_adjust(left=0.2, right=0.95, top=0.76, bottom=0.12)
names = [n for _, n in CONFIGS]
y = range(len(CONFIGS))
h = 0.36
f_all = [metrics[k]["all"]["failure_f1"] for k, _ in CONFIGS]
f_ho = [metrics[k]["heldout"]["failure_f1"] for k, _ in CONFIGS]
ax.barh([i - h / 2 - 0.02 for i in y], f_all, height=h, color=BLUE, label="All 87 sessions")
ax.barh([i + h / 2 + 0.02 for i in y], f_ho, height=h, color=ORANGE, label="Held-out 15 sessions")
for i, (a, b) in enumerate(zip(f_all, f_ho)):
    ax.text(a + 0.01, i - h / 2 - 0.02, f"{a:.2f}", va="center", fontsize=10, color=INK)
    ax.text(b + 0.01, i + h / 2 + 0.02, f"{b:.2f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(list(y), names)
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("Failure-detection F1 (precision is 1.00 in every configuration)")
_clean(ax)
fig.legend(loc="upper left", bbox_to_anchor=(0.2, 0.86), ncol=2, frameon=False, fontsize=10)
_title(fig, "Hybrid 120B catches the most failures", "F1 on all sessions and on the held-out paraphrased and French sessions")
fig.savefig(OUT / "f1_by_config.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 2. recall per failure type heatmap
types = list(FAILURE_CODES)
recall = defaultdict(dict)
counts = {t: sum(g["expected_failure_type"] == t for g in gt.values()) for t in types}
for key, _ in CONFIGS:
    an = {a.session_id: a for a in store.load_analyses(key)}
    for t in types:
        rows = [sid for sid, g in gt.items() if g["expected_failure_type"] == t]
        recall[key][t] = sum(an[s].outcome == "failure" for s in rows) / len(rows)
fig, ax = plt.subplots(figsize=(10, 5.2))
fig.subplots_adjust(left=0.36, right=0.97, top=0.8, bottom=0.08)
cmap = LinearSegmentedColormap.from_list("seq", SEQ)
data = [[recall[k][t] for k, _ in CONFIGS] for t in types]
ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i, row in enumerate(data):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10.5, color="#ffffff" if v > 0.55 else INK)
ax.set_xticks(range(len(CONFIGS)), [n.replace(" · ", "\n") for n in names], fontsize=10)
ax.set_yticks(range(len(types)), [f"{FAILURE_CODES[t]} {FAILURE_LABELS[t]} (n={counts[t]})" for t in types], fontsize=10)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks([x - 0.5 for x in range(1, len(CONFIGS))], minor=True)
ax.set_yticks([x - 0.5 for x in range(1, len(types))], minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
_title(fig, "Rules and LLM cover different failure types", "Recall per failure type: share of true failures flagged as failures")
fig.savefig(OUT / "recall_by_type.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 3. cost vs F1
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.subplots_adjust(left=0.12, right=0.95, top=0.78, bottom=0.14)
xs = [metrics[k]["all"]["cost_usd_per_1k_sessions"] for k, _ in CONFIGS]
ys = [metrics[k]["all"]["failure_f1"] for k, _ in CONFIGS]
ax.scatter(xs, ys, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
offsets = {"Rules only": (8, -4), "LLM only · 20B": (8, -12), "Hybrid · 20B": (8, 6),
           "LLM only · 120B": (-118, -4), "Hybrid · 120B": (-104, 6)}
for x, yv, n in zip(xs, ys, names):
    ax.annotate(f"{n}  {yv:.2f}", (x, yv), textcoords="offset points", xytext=offsets[n], fontsize=10, color=INK)
ax.set_xlabel("Estimated LLM cost, USD per 1,000 sessions (Groq list prices)")
ax.set_ylabel("Failure F1")
ax.set_ylim(0.7, 1.0)
ax.set_xlim(-0.03, 0.46)
_clean(ax, "both")
_title(fig, "Better detection costs about $0.40 per 1,000 sessions", "Failure F1 against estimated judge cost")
fig.savefig(OUT / "cost_vs_f1.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- 4. ranked issues
issues = store.load_issues("hybrid:openai/gpt-oss-120b")[:10]
fig, ax = plt.subplots(figsize=(11, 5.4))
fig.subplots_adjust(left=0.45, right=0.93, top=0.8, bottom=0.1)
labels = [f"#{i['rank']} {i['code']} {i['title']}" for i in issues]
pr = [i["priority"] for i in issues]
ax.barh(range(len(issues)), pr, height=0.62, color=BLUE)
for k, i in enumerate(issues):
    ax.text(i["priority"] + 0.3, k, f"{i['priority']:.1f}  ·  {i['affected_sessions']} sessions · {i['impact']} impact",
            va="center", fontsize=9.5, color=INK2)
ax.set_yticks(range(len(issues)), labels, fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, max(pr) * 1.55)
ax.set_xlabel("Priority = affected sessions × impact weight × mean confidence")
_clean(ax)
_title(fig, "What to investigate first", "Top 10 issue groups found by the hybrid 120B analyzer")
fig.savefig(OUT / "ranked_issues.png", dpi=160)
plt.close(fig)

print("charts written to", OUT)
