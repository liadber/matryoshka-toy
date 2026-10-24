"""
Matryoshka Language Model Suites — a toy reimplementation.
Paper: Godey & Artzi, 2026 (arXiv 2608.09703). Course: Active AI Reading, week 2.

What this does
--------------
Trains, on a tiny character-level text, two things:
  1. VANILLA  : three independent transformers (small / medium / large).
  2. MATRYOSHKA: ONE transformer with three exit points, where the small model is
                 the first layers of the medium, which is the first layers of the large
                 (same weights, narrower). Trained end-to-end with one loss.
Then it plots the validation loss of each of the six models over training, counts
parameters, and measures wall-clock time.

It reproduces the MECHANISM and the SHAPE of the result, not the paper's numbers.

Run:  python matryoshka_toy.py            (≈ 3–6 minutes on a laptop CPU)
      python matryoshka_toy.py --no-rescale   (ablation: junction without norm matching)
      python matryoshka_toy.py --alpha 0      (ablation: no distillation)
"""
import argparse, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1500, help="training steps per model")
p.add_argument("--alpha", type=float, default=0.3, help="distillation weight (paper: 0.3)")
p.add_argument("--no-rescale", action="store_true", help="ablation: skip norm matching in the junction")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)

# The suite. Each block ADDS layers on top of the previous block, at a wider width.
# (paper: 24 layers @1024  →  +10 @2304  →  +5 @4352.  Ours is a toy.)
SUITE = [  # (extra layers, width)
    (2, 64),   # exit 1: "small"  = 2 layers @ 64
    (2, 128),  # exit 2: "medium" = small + 2 layers @ 128
    (2, 256),  # exit 3: "large"  = medium + 2 layers @ 256
]
NAMES = ["small", "medium", "large"]
CTX = 48          # context length (characters)
BATCH = 32
LR = 3e-3
EVAL_EVERY = 50

# ----------------------------------------------------------------------------
# 1. A toy corpus: sentences from templates, character-level
# ----------------------------------------------------------------------------
SUBJ = ["the cat", "a dog", "my friend", "the old man", "a small bird", "the teacher", "our neighbor"]
VERB = ["sees", "likes", "finds", "watches", "follows", "needs", "remembers"]
OBJ  = ["the river", "a red car", "the moon", "a warm house", "the garden", "a long road", "the market"]
TAIL = ["in the morning", "after the rain", "every day", "at night", "in winter", "on sunday", "before noon"]

def make_corpus(n=4000):
    s = []
    for _ in range(n):
        s.append(f"{random.choice(SUBJ)} {random.choice(VERB)} {random.choice(OBJ)} {random.choice(TAIL)}.")
    return " ".join(s)

text = make_corpus()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CTX - 1, (BATCH,))
    x = torch.stack([d[i:i + CTX] for i in ix])
    y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
    return x, y

# ----------------------------------------------------------------------------
# 2. A minimal transformer block (pre-norm, causal attention, MLP)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)   # causal: see only the past
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class Stack(nn.Module):
    """n blocks at width d, with an input embedding (token + position) at that width."""
    def __init__(self, n_layers, d):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.Sequential(*[Block(d) for _ in range(n_layers)])

    def embed(self, idx):
        T = idx.size(1)
        return self.tok(idx) + self.pos(torch.arange(T))

# ----------------------------------------------------------------------------
# 3. VANILLA: one independent model per size (the baseline)
# ----------------------------------------------------------------------------
class Vanilla(nn.Module):
    def __init__(self, n_layers, d):
        super().__init__()
        self.stack = Stack(n_layers, d)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        x = self.stack.embed(idx)
        x = self.stack.blocks(x)
        return self.head(self.ln_f(x))

# ----------------------------------------------------------------------------
# 4. MATRYOSHKA: one stack, three exits, junctions in between
# ----------------------------------------------------------------------------
class Matryoshka(nn.Module):
    def __init__(self, suite, rescale=True):
        super().__init__()
        self.rescale = rescale
        self.stacks, self.ln_f, self.heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for n_layers, d in suite:
            self.stacks.append(Stack(n_layers, d))      # block m: its own layers + a FRESH embedding at width d_m
            self.ln_f.append(nn.LayerNorm(d))           # one final norm per exit
            self.heads.append(nn.Linear(d, V, bias=False))  # one LM head per exit  (paper: W_m)

    def forward(self, idx):
        """Returns a list of logits, one per exit, from ONE forward pass."""
        logits = []
        x = None
        for m, stack in enumerate(self.stacks):
            e = stack.embed(idx)                        # e^{m}_θ(x): fresh embedding at width d_m
            if x is None:
                h = e                                   # block 1 starts from the embedding
            else:
                # ---- the inter-model junction (paper, section 2) ----
                d_prev = x.size(-1)
                if self.rescale:
                    # õ^m = o^m · ‖e^{m+1}‖ / ‖o^m‖   (match the norm of a fresh embedding)
                    e_new = e[..., d_prev:]             # the NEW dimensions only
                    x = x * (e_new.norm(dim=-1, keepdim=True) / (x.norm(dim=-1, keepdim=True) + 1e-6))
                h = torch.cat([x, e[..., d_prev:]], dim=-1)   # o^{m+1} input = concat(õ^m, e^{m+1})
            x = stack.blocks(h)                         # the extra layers of block m
            logits.append(self.heads[m](self.ln_f[m](x)))   # this exit's table of logits
        return logits

def matryoshka_loss(logits_list, y, alpha):
    """Σ_m [ (1−α)·CE_m + α·KD_m ] ;  KD from the LARGEST exit, with stop_grad on the teacher."""
    teacher = logits_list[-1].detach()                      # stop_grad(σ(l^M))
    p_teacher = F.softmax(teacher, dim=-1)
    total, parts = 0.0, []
    for m, lg in enumerate(logits_list):
        ce = F.cross_entropy(lg.view(-1, V), y.view(-1))    # −log p_m(correct char)
        if m < len(logits_list) - 1 and alpha > 0:
            kd = -(p_teacher * F.log_softmax(lg, dim=-1)).sum(-1).mean()   # Σ_v p_M(v)·(−log p_m(v))
            loss_m = (1 - alpha) * ce + alpha * kd
        else:
            loss_m = ce                                     # the largest model has no teacher
        total = total + loss_m
        parts.append(ce.item())
    return total, parts

# ----------------------------------------------------------------------------
# 5. Training loop (shared by both setups)
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_loss(fn, n=10):
    out = []
    for _ in range(n):
        x, y = get_batch("val")
        out.append(fn(x, y))
    return [sum(c) / n for c in zip(*out)]

def train(model, step_fn, eval_fn, steps, label):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    curve, t0 = [], time.time()
    for it in range(steps + 1):
        if it % EVAL_EVERY == 0:
            model.eval(); curve.append((it, eval_loss(eval_fn))); model.train()
        if it == steps: break
        x, y = get_batch("train")
        loss = step_fn(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    secs = time.time() - t0
    print(f"{label:<28} final val loss: " + "  ".join(f"{c:.3f}" for c in curve[-1][1]) + f"   ({secs:.0f}s)")
    return curve, secs

def n_params(m): return sum(p.numel() for p in m.parameters())

# ----------------------------------------------------------------------------
# 6. Run everything
# ----------------------------------------------------------------------------
print(f"corpus: {len(text):,} chars, vocab {V}, ctx {CTX}, steps {args.steps}, alpha {args.alpha}, rescale {not args.no_rescale}\n")

# --- Vanilla: three separate models -----------------------------------------
vanilla_curves, vanilla_secs, vanilla_params = {}, 0.0, 0
total_layers, _ = 0, 0
for (n_layers, d), name in zip(SUITE, NAMES):
    total_layers += n_layers
    vm = Vanilla(total_layers, d)       # a vanilla model with the SAME depth and width as that exit
    vanilla_params += n_params(vm)
    curve, secs = train(
        vm,
        step_fn=lambda x, y, vm=vm: F.cross_entropy(vm(x).view(-1, V), y.view(-1)),
        eval_fn=lambda x, y, vm=vm: [F.cross_entropy(vm(x).view(-1, V), y.view(-1)).item()],
        steps=args.steps, label=f"vanilla {name} ({total_layers}L@{d})")
    vanilla_curves[name] = [(it, c[0]) for it, c in curve]
    vanilla_secs += secs

# --- Matryoshka: one model, three exits -------------------------------------
mm = Matryoshka(SUITE, rescale=not args.no_rescale)
curve, matry_secs = train(
    mm,
    step_fn=lambda x, y: matryoshka_loss(mm(x), y, args.alpha)[0],
    eval_fn=lambda x, y: matryoshka_loss(mm(x), y, 0.0)[1],      # report plain CE per exit
    steps=args.steps, label="matryoshka (3 exits)")
matry_curves = {name: [(it, c[m]) for it, c in curve] for m, name in enumerate(NAMES)}

# ----------------------------------------------------------------------------
# 7. Report: parameters, time, final gap
# ----------------------------------------------------------------------------
mp = n_params(mm)
print(f"\nparameters: vanilla suite {vanilla_params:,}  |  matryoshka {mp:,}  →  {100*(1-mp/vanilla_params):.0f}% fewer")
print(f"train time: vanilla suite {vanilla_secs:.0f}s  |  matryoshka {matry_secs:.0f}s  →  {100*(1-matry_secs/vanilla_secs):.0f}% less")
print("\nper-size validation loss (lower is better):")
print(f"{'':8} {'vanilla':>9} {'matryoshka':>11} {'gap':>8}")
for name in NAMES:
    v, m = vanilla_curves[name][-1][1], matry_curves[name][-1][1]
    print(f"{name:8} {v:9.3f} {m:11.3f} {100*(m-v)/v:+7.1f}%")

# ----------------------------------------------------------------------------
# 8. Plot
# ----------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"small": "#7A3E1D", "medium": "#A8663F", "large": "#1D4F7A"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in NAMES:
        xs, ys = zip(*vanilla_curves[name]); ax.plot(xs, ys, "--", color=colors[name], label=f"vanilla {name}")
        xs, ys = zip(*matry_curves[name]);   ax.plot(xs, ys, "-",  color=colors[name], label=f"matryoshka {name}")
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss (cross-entropy, chars)")
    ax.set_title(f"Three separate models (dashed) vs one nested model with three exits (solid)\n"
                 f"params: {vanilla_params:,} vs {mp:,} ({100*(1-mp/vanilla_params):.0f}% fewer)   "
                 f"alpha={args.alpha}  rescale={not args.no_rescale}", fontsize=10)
    ax.legend(); ax.grid(alpha=.3)
    tag = ("_norescale" if args.no_rescale else "") + (f"_alpha{args.alpha}" if args.alpha != 0.3 else "")
    out = f"curves{tag}.png"; fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")
except ImportError:
    print("\n(matplotlib not installed — skipping the plot)")
