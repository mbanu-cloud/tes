"""Figure 1 (DASI-MSNet as implemented)."""
import matplotlib, os
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'); os.makedirs(FIG, exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ink, sec, mut = '#0b0b0b', '#52514e', '#898781'
BLUE, BLUE_L = '#2a78d6', '#e8f1fc'
RED, RED_L = '#c8462f', '#fdece8'
GREY_L, GREEN_L, GREEN = '#f3f3f1', '#e6f6ef', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

W = 7.5; H = W / 1.828
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 54.7); ax.axis('off')

def box(x, y, w, h, title, lines, fc, ec, tsize=6.6, lsize=5.6, dashed=False, sp=2.45):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8', fc=fc, ec=ec,
                                lw=0.9, ls=(0, (3, 2)) if dashed else '-'))
    n = len(lines); top = y + h - 2.2
    ax.text(x + w / 2, top, title, ha='center', va='top', fontsize=tsize, fontweight='bold', color=ink)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, top - 3.0 - i * sp, l, ha='center', va='top', fontsize=lsize, color=sec)

def arrow(x1, y1, x2, y2, color=sec, ls='-', rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, color=color, lw=lw,
                                 ls=ls, connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1))

# 1. input
box(1, 17, 15.5, 27, 'Input sequence', ['X ∈ ℝ^(10 × 44)', 'window T = 10 frames,', 'stride 3 frames', '', '16 FACS AUs (MediaPipe)', '12 gaze + head pose', '16 AU velocities'], GREY_L, mut)
# 2. projection
box(20, 24.5, 11.5, 12, 'Input projection', ['Linear', '44 → 64'], GREY_L, mut)
arrow(16.8, 30.5, 19.8, 30.5)
# 3. multi-scale block container
ax.add_patch(FancyBboxPatch((35, 9.5), 29.5, 41, boxstyle='round,pad=0.25,rounding_size=1.0', fc='white', ec=GREEN, lw=1.0, ls=(0, (4, 2))))
ax.text(49.75, 48.6, 'Multi-scale temporal block  (× 2, stacked)', ha='center', va='top', fontsize=6.6, fontweight='bold', color=ink)
br = [('Micro', 'Conv1d k = 3, d = 1', '21 channels'), ('Meso', 'Conv1d k = 5, d = 1', '21 channels'), ('Macro', 'Conv1d k = 7, d = 2', '22 channels')]
ys = [36.3, 25.3, 14.3]
for (t, l1, l2), yb in zip(br, ys):
    box(36.5, yb, 13.5, 9, t, [l1, l2], GREEN_L, GREEN, tsize=6.2, lsize=5.4, sp=2.1)
    arrow(32, 30.5, 36.3, yb + 4.5, rad=0.0)
    arrow(50.2, yb + 4.5, 53.3, 30.5)
box(53.5, 21, 9.8, 19, 'Merge', ['concat (64)', '+ residual x', 'LayerNorm', 'GELU'], GREEN_L, GREEN, tsize=6.2, lsize=5.4)
# 4. attention pooling
box(67.5, 21, 12, 19, 'Attention pooling', ['α_t = softmax(', 'vᵀ tanh(W H_t))', '', 'z = Σ α_t H_t', 'z ∈ ℝ^64'], GREY_L, mut, lsize=5.4)
arrow(63.6, 30.5, 67.3, 30.5)
# 5a. engagement head
box(82.5, 31, 17, 20, 'Engagement head', ['MLP 64 → 32 → 3', '(GELU, dropout 0.2)', '', 'L_task: ordinal focal loss', 'γ = 2, cost matrix C,', 'no class weights'], BLUE_L, BLUE, lsize=5.3)
arrow(79.8, 33.5, 82.3, 41, color=BLUE, lw=1.1)
# 5b. GRL + subject head
box(82.5, 2.5, 17, 25.5, 'Subject discriminator', ['GRL R(z):', 'forward z, backward −λ', '(λ = 0.5)', '', 'MLP 64 → 64 → S_train', '(LayerNorm, GELU, dropout 0.3)', 'L_subj: cross-entropy,', 'weight μ = 0.5'], RED_L, RED, lsize=5.0, sp=2.3)
arrow(79.8, 27.5, 82.3, 20, color=RED, lw=1.1)
# reversed gradient back to encoder
arrow(82.3, 8.5, 50, 9.2, color=RED, ls=(0, (3, 2)), rad=-0.12, lw=0.9)
ax.text(68, 4.9, 'reversed gradient (× −λ) to the encoder', ha='center', va='center', fontsize=5.4, color=RED, style='italic')
# total loss & finding
ax.text(1, 11.5, 'L_total = L_task + μ · L_subj', ha='left', va='center', fontsize=6.2, color=ink, fontweight='bold')
ax.add_patch(FancyBboxPatch((1, 1.2), 31.5, 7.3, boxstyle='round,pad=0.25,rounding_size=0.8', fc='#fff8e6', ec='#eda100', lw=0.9))
ax.text(16.75, 6.6, 'Negative control (this study)', ha='center', va='center', fontsize=5.9, fontweight='bold', color=ink)
ax.text(16.75, 3.4, 'Identity stays linearly decodable from z:\nprobe 28.6% vs. 4.55% chance (−0.97 pp vs. λ = 0)', ha='center', va='center', fontsize=5.1, color=sec, linespacing=1.2)

fig.savefig(os.path.join(FIG, 'fig1_dasi_msnet.png'), dpi=400, facecolor='white')
