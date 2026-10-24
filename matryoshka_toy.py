"""
Matryoshka Language Model Suites — a toy reimplementation.
Paper: Godey & Artzi, 2026 (arXiv 2608.09703). Course: Active AI Reading, week 2.

What this does
--------------
Trains, on a small character-level text, two things:
  1. VANILLA    : three independent transformers (small / medium / large), each with the
                  same depth as the matching nested exit and a width chosen so that its
                  parameter count matches that exit's cumulative count (as in the paper's Table 1).
  2. MATRYOSHKA : ONE transformer with three exit points, where the small model is the
                  first layers of the medium, which is the first layers of the large
                  (same weights, narrower). Trained end-to-end with one loss.
Then it reports, for each size: validation loss over training, parameter counts,
wall-clock time, and how often two models agree on the next character (paper, 3.2.2).

It reproduces the MECHANISM and the SHAPE of the result, not the paper's numbers.

Run:  python matryoshka_toy.py                 (about 15 minutes on a laptop CPU; run ONE at a time)
      python matryoshka_toy.py --no-rescale    (ablation: junction without norm matching)
      python matryoshka_toy.py --alpha 0       (ablation: no distillation)
"""
import argparse, json, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. Settings
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1500, help="training steps per model")
parser.add_argument("--alpha", type=float, default=0.3, help="distillation weight (paper: 0.3)")
parser.add_argument("--no-rescale", action="store_true", help="ablation: skip norm matching in the junction")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)

# The suite. Each block ADDS layers on top of the previous block, at a wider width.
# (paper: 24 layers @1024  →  +10 @2304  →  +5 @4352.  Ours is a toy.)
SUITE = [   # (extra layers, width)
    (2, 64),    # exit 1: "small"  = 2 layers @ 64
    (2, 128),   # exit 2: "medium" = small  + 2 layers @ 128
    (2, 256),   # exit 3: "large"  = medium + 2 layers @ 256
]
SIZE_NAMES = ["small", "medium", "large"]
CONTEXT_LENGTH = 96      # characters the model sees at once: long enough to hold a whole sentence
BATCH_SIZE = 32
LEARNING_RATE = 2e-3     # with warmup + gradient clipping (v1 used 3e-3 without; the 6-layer vanilla did not converge)
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
EVAL_EVERY = 50
ATTENTION_HEADS = 4

# ----------------------------------------------------------------------------
# 1. A toy corpus: sentences from a small grammar, character-level
# ----------------------------------------------------------------------------
# Harder than fixed templates: bigger vocabulary, singular/plural AGREEMENT across a relative
# clause ("the cats that the man feeds sleep"), and a NAME that must be copied from the start of
# the sentence to its end. Short-range patterns are easy; agreement and the copy need depth/attention.
PERSON_NAMES = ["anna", "omer", "dana", "yoav", "noa", "tal", "liad", "maya", "eli", "roni", "gal", "shir"]
NOUNS = ["cat", "dog", "bird", "child", "teacher", "farmer", "sailor", "doctor", "baker", "driver",
         "horse", "fox", "rabbit", "painter", "singer", "soldier", "student", "nurse", "pilot", "cook"]
ADJECTIVES = ["old", "small", "tired", "happy", "quiet", "brave", "clever", "lazy", "young", "tall", "", "", ""]
TRANSITIVE_VERBS = [("sees", "see"), ("likes", "like"), ("feeds", "feed"), ("follows", "follow"), ("calls", "call"),
                    ("needs", "need"), ("helps", "help"), ("finds", "find"), ("watches", "watch"), ("knows", "know")]
INTRANSITIVE_VERBS = [("sleeps", "sleep"), ("waits", "wait"), ("runs", "run"), ("sings", "sing"), ("laughs", "laugh"),
                      ("rests", "rest"), ("hides", "hide"), ("smiles", "smile")]
PLACES = ["in the garden", "by the river", "on the roof", "near the market", "at the station", "under the tree",
          "in the kitchen", "behind the house", "on the hill", "at the harbor"]

def noun_phrase(plural):
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS) + ("s" if plural else "")
    return "the " + (adjective + " " if adjective else "") + noun

def sentence():
    name = random.choice(PERSON_NAMES)
    subject_is_plural = random.random() < 0.5
    parts = [name + " says that", noun_phrase(subject_is_plural)]
    if random.random() < 0.5:                                   # a relative clause between subject and verb
        clause_is_plural = random.random() < 0.5
        clause_verb = random.choice(TRANSITIVE_VERBS)[1 if clause_is_plural else 0]
        parts.append("that " + noun_phrase(clause_is_plural) + " " + clause_verb)
    main_verb = random.choice(INTRANSITIVE_VERBS)[1 if subject_is_plural else 0]   # agrees with the SUBJECT, not the clause
    parts.append(main_verb)
    parts.append(random.choice(PLACES))
    if random.random() < 0.7:
        parts.append(", said " + name)                          # copy the name from the start
    return " ".join(parts) + "."

def make_corpus(n_sentences=6000):
    return " ".join(sentence() for _ in range(n_sentences))

text = make_corpus()
alphabet = sorted(set(text))
VOCAB_SIZE = len(alphabet)                                      # paper: V
char_to_id = {ch: i for i, ch in enumerate(alphabet)}
data = torch.tensor([char_to_id[ch] for ch in text], dtype=torch.long)
n_train = int(0.9 * len(data))
train_data, val_data = data[:n_train], data[n_train:]

def get_batch(split):
    """BATCH_SIZE random windows of CONTEXT_LENGTH characters; targets are the same windows shifted by one."""
    source = train_data if split == "train" else val_data
    starts = torch.randint(len(source) - CONTEXT_LENGTH - 1, (BATCH_SIZE,))
    inputs = torch.stack([source[s:s + CONTEXT_LENGTH] for s in starts])
    targets = torch.stack([source[s + 1:s + CONTEXT_LENGTH + 1] for s in starts])
    return inputs, targets

# ----------------------------------------------------------------------------
# 2. A minimal transformer block (pre-norm, causal attention, MLP)
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm_before_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, ATTENTION_HEADS, batch_first=True)
        self.norm_before_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, hidden):
        seq_len = hidden.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1)   # each position sees only the past
        normed = self.norm_before_attention(hidden)
        hidden = hidden + self.attention(normed, normed, normed, attn_mask=causal_mask, need_weights=False)[0]
        hidden = hidden + self.mlp(self.norm_before_mlp(hidden))
        return hidden

class LayerStack(nn.Module):
    """n_layers transformer blocks at one width, with an input embedding (token + position) at that width."""
    def __init__(self, n_layers, width):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, width)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, width)
        self.blocks = nn.Sequential(*[TransformerBlock(width) for _ in range(n_layers)])

    def embed(self, token_ids):                                 # paper: e_θ(x)
        seq_len = token_ids.size(1)
        return self.token_embedding(token_ids) + self.position_embedding(torch.arange(seq_len))

# ----------------------------------------------------------------------------
# 3. VANILLA: one independent model per size (the baseline)
# ----------------------------------------------------------------------------
class VanillaLM(nn.Module):
    def __init__(self, n_layers, width):
        super().__init__()
        self.stack = LayerStack(n_layers, width)
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, VOCAB_SIZE, bias=False)      # paper: W

    def forward(self, token_ids):
        hidden = self.stack.embed(token_ids)
        hidden = self.stack.blocks(hidden)
        return self.lm_head(self.final_norm(hidden))                 # logits: one score per character

# ----------------------------------------------------------------------------
# 4. MATRYOSHKA: one stack, three exits, a junction between consecutive blocks
# ----------------------------------------------------------------------------
class MatryoshkaLM(nn.Module):
    def __init__(self, suite, rescale=True):
        super().__init__()
        self.rescale = rescale
        self.stacks, self.final_norms, self.lm_heads = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for n_layers, width in suite:
            self.stacks.append(LayerStack(n_layers, width))          # block m: its own layers + a FRESH embedding at width_m
            self.final_norms.append(nn.LayerNorm(width))             # one final norm per exit
            self.lm_heads.append(nn.Linear(width, VOCAB_SIZE, bias=False))   # one LM head per exit (paper: W_m)

    def forward(self, token_ids):
        """Returns a list of logits, one per exit, from ONE forward pass."""
        logits_per_exit = []
        hidden = None                                               # output of the previous block (paper: o^m)
        for exit_index, stack in enumerate(self.stacks):
            fresh_embedding = stack.embed(token_ids)                # paper: e^{m}_θ(x), at this block's width
            if hidden is None:
                block_input = fresh_embedding                       # block 1 starts from the embedding
            else:
                # ---- the inter-model junction (paper, section 2) ----
                previous_width = hidden.size(-1)
                new_dimensions = fresh_embedding[..., previous_width:]      # only the dimensions this block ADDS
                if self.rescale:
                    # õ^m = o^m · ‖e^{m+1}‖ / ‖o^m‖ : make the previous output as "long" as a fresh embedding
                    hidden = hidden * (new_dimensions.norm(dim=-1, keepdim=True)
                                       / (hidden.norm(dim=-1, keepdim=True) + 1e-6))
                block_input = torch.cat([hidden, new_dimensions], dim=-1)   # concat(õ^m, e^{m+1})
            hidden = stack.blocks(block_input)                      # the extra layers of this block
            logits_per_exit.append(self.lm_heads[exit_index](self.final_norms[exit_index](hidden)))
        return logits_per_exit

def matryoshka_loss(logits_per_exit, targets, alpha):
    """Σ_m [ (1−α)·CE_m + α·KD_m ] ;  KD from the LARGEST exit, with stop_grad on the teacher.
    Returns the total loss (to train on) and the plain cross-entropy of each exit (to report)."""
    teacher_logits = logits_per_exit[-1].detach()                   # stop_grad(σ(l^M)): the teacher is a constant here
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    total_loss, cross_entropies = 0.0, []
    for exit_index, logits in enumerate(logits_per_exit):
        cross_entropy = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))   # −log p_m(correct char)
        is_largest = exit_index == len(logits_per_exit) - 1
        if not is_largest and alpha > 0:
            distillation = -(teacher_probs * F.log_softmax(logits, dim=-1)).sum(-1).mean()   # Σ_v p_M(v)·(−log p_m(v))
            exit_loss = (1 - alpha) * cross_entropy + alpha * distillation
        else:
            exit_loss = cross_entropy                               # the largest exit has no teacher
        total_loss = total_loss + exit_loss
        cross_entropies.append(cross_entropy.item())
    return total_loss, cross_entropies

# ----------------------------------------------------------------------------
# 5. Training loop (shared by both setups)
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(loss_fn, n_batches=10):
    """Average of loss_fn over a few validation batches. loss_fn returns a list (one number per exit)."""
    per_batch = [loss_fn(*get_batch("val")) for _ in range(n_batches)]
    return [sum(values) / n_batches for values in zip(*per_batch)]

def train(model, train_loss_fn, eval_loss_fn, steps, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / WARMUP_STEPS))
    curve, start_time = [], time.time()
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0:
            model.eval(); curve.append((step, evaluate(eval_loss_fn))); model.train()
        if step == steps:
            break
        inputs, targets = get_batch("train")
        loss = train_loss_fn(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); warmup.step()
    seconds = time.time() - start_time
    print(f"{label:<40} final val loss: " + "  ".join(f"{v:.3f}" for v in curve[-1][1]) + f"   ({seconds:.0f}s)")
    return curve, seconds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# ----------------------------------------------------------------------------
# 6. Run everything
# ----------------------------------------------------------------------------
print(f"corpus: {len(text):,} chars, vocab {VOCAB_SIZE}, context {CONTEXT_LENGTH}, steps {args.steps}, "
      f"alpha {args.alpha}, rescale {not args.no_rescale}\n")

# --- The nested model first (we need its per-exit parameter counts to size the baselines) ---
matryoshka = MatryoshkaLM(SUITE, rescale=not args.no_rescale)

def nested_parameters_up_to(exit_index):
    """Parameters a user of this exit actually runs: blocks 0..exit, plus that exit's norm and head (paper: 'Cumul.')."""
    return (sum(count_parameters(matryoshka.stacks[i]) for i in range(exit_index + 1))
            + count_parameters(matryoshka.final_norms[exit_index]) + count_parameters(matryoshka.lm_heads[exit_index]))

def width_matching(n_layers, target_parameters):
    """Smallest width (multiple of ATTENTION_HEADS) whose vanilla model has at least target_parameters."""
    width = 16
    while count_parameters(VanillaLM(n_layers, width)) < target_parameters:
        width += ATTENTION_HEADS
    return width

# --- Vanilla: three separate models, parameter-matched to each exit ---
vanilla_curves, vanilla_models, vanilla_shapes = {}, {}, {}
vanilla_seconds, vanilla_parameters = 0.0, 0
depth_so_far = 0
for exit_index, ((extra_layers, width), size) in enumerate(zip(SUITE, SIZE_NAMES)):
    depth_so_far += extra_layers
    target = nested_parameters_up_to(exit_index)
    matched_width = width_matching(depth_so_far, target)          # same depth as the exit, width chosen to match its parameters
    vanilla = VanillaLM(depth_so_far, matched_width)
    vanilla_parameters += count_parameters(vanilla)
    vanilla_shapes[size] = (depth_so_far, matched_width, count_parameters(vanilla), target)

    def vanilla_train_loss(inputs, targets, model=vanilla):
        return F.cross_entropy(model(inputs).view(-1, VOCAB_SIZE), targets.view(-1))

    def vanilla_eval_loss(inputs, targets, model=vanilla):
        return [F.cross_entropy(model(inputs).view(-1, VOCAB_SIZE), targets.view(-1)).item()]

    curve, seconds = train(vanilla, vanilla_train_loss, vanilla_eval_loss, args.steps,
                           label=f"vanilla {size} ({depth_so_far}L@{matched_width}, {count_parameters(vanilla)/1e6:.2f}M)")
    vanilla_curves[size] = [(step, values[0]) for step, values in curve]
    vanilla_seconds += seconds
    vanilla_models[size] = vanilla

# --- Matryoshka: one model, three exits ---
def matryoshka_train_loss(inputs, targets):
    return matryoshka_loss(matryoshka(inputs), targets, args.alpha)[0]

def matryoshka_eval_loss(inputs, targets):
    return matryoshka_loss(matryoshka(inputs), targets, 0.0)[1]    # report plain cross-entropy per exit

curve, matryoshka_seconds = train(matryoshka, matryoshka_train_loss, matryoshka_eval_loss, args.steps,
                                  label=f"matryoshka (3 exits, {count_parameters(matryoshka)/1e6:.2f}M)")
matryoshka_curves = {size: [(step, values[exit_index]) for step, values in curve]
                     for exit_index, size in enumerate(SIZE_NAMES)}

# ----------------------------------------------------------------------------
# 7. Report: parameters, time, quality per size, agreement
# ----------------------------------------------------------------------------
matryoshka_parameters = count_parameters(matryoshka)
print(f"\nparameters: vanilla suite {vanilla_parameters:,}  |  matryoshka {matryoshka_parameters:,}"
      f"  →  {100*(1-matryoshka_parameters/vanilla_parameters):.0f}% fewer")
print(f"train time: vanilla suite {vanilla_seconds:.0f}s  |  matryoshka {matryoshka_seconds:.0f}s"
      f"  →  {100*(1-matryoshka_seconds/vanilla_seconds):.0f}% less")

print("\nshapes (vanilla matched to the nested exit's cumulative parameters):")
depth_so_far = 0
for exit_index, ((extra_layers, width), size) in enumerate(zip(SUITE, SIZE_NAMES)):
    depth_so_far += extra_layers
    depth, matched_width, vanilla_count, nested_count = vanilla_shapes[size]
    print(f"  {size:7} vanilla {depth}L@{matched_width} = {vanilla_count/1e6:.2f}M   |   "
          f"nested exit {depth_so_far}L, widths {[w for _, w in SUITE[:exit_index+1]]}, cumul. {nested_count/1e6:.2f}M")

print("\nper-size validation loss (lower is better):")
print(f"{'':8} {'vanilla':>9} {'matryoshka':>11} {'gap':>8}")
results = {"settings": vars(args), "vanilla_parameters": vanilla_parameters, "matryoshka_parameters": matryoshka_parameters,
           "vanilla_seconds": vanilla_seconds, "matryoshka_seconds": matryoshka_seconds, "per_size": {}}
for size in SIZE_NAMES:
    vanilla_final, matryoshka_final = vanilla_curves[size][-1][1], matryoshka_curves[size][-1][1]
    gap_pct = 100 * (matryoshka_final - vanilla_final) / vanilla_final
    print(f"{size:8} {vanilla_final:9.3f} {matryoshka_final:11.3f} {gap_pct:+7.1f}%")
    results["per_size"][size] = {"vanilla": vanilla_final, "matryoshka": matryoshka_final, "gap_pct": gap_pct,
                                 "vanilla_curve": vanilla_curves[size], "matryoshka_curve": matryoshka_curves[size]}

# --- Agreement (paper, 3.2.2): how often two models pick the SAME next character ---
# Speculative decoding accepts a draft token when the verifier would have chosen it too,
# so draft/verifier agreement is what turns a small model into a useful draft.
@torch.no_grad()
def agreement_matrix(predict_fns, n_batches=20):
    """predict_fns: list of fn(inputs) -> predicted next character ids [B, T]. Returns pairwise agreement rates."""
    k = len(predict_fns)
    agreement = [[0.0] * k for _ in range(k)]
    for _ in range(n_batches):
        inputs, _ = get_batch("val")
        predictions = [fn(inputs) for fn in predict_fns]
        for i in range(k):
            for j in range(k):
                agreement[i][j] += (predictions[i] == predictions[j]).float().mean().item() / n_batches
    return agreement

matryoshka.eval()
for model in vanilla_models.values():
    model.eval()
nested_predictors = [lambda inputs, m=m: matryoshka(inputs)[m].argmax(-1) for m in range(len(SIZE_NAMES))]
vanilla_predictors = [lambda inputs, model=vanilla_models[size]: model(inputs).argmax(-1) for size in SIZE_NAMES]
agreement_nested = agreement_matrix(nested_predictors)
agreement_vanilla = agreement_matrix(vanilla_predictors)

print("\nagreement on the next character (draft vs verifier; what speculative decoding needs):")
print(f"{'pair':16} {'vanilla':>9} {'matryoshka':>11}")
PAIRS = [(0, 1), (1, 2), (0, 2)]
results["agreement"] = {}
for i, j in PAIRS:
    pair = f"{SIZE_NAMES[i]}/{SIZE_NAMES[j]}"
    print(f"{pair:16} {100*agreement_vanilla[i][j]:8.1f}% {100*agreement_nested[i][j]:10.1f}%")
    results["agreement"][pair] = {"vanilla": agreement_vanilla[i][j], "matryoshka": agreement_nested[i][j]}

run_tag = (("_norescale" if args.no_rescale else "") + (f"_alpha{args.alpha}" if args.alpha != 0.3 else "")
           + (f"_seed{args.seed}" if args.seed else ""))
with open(f"results{run_tag}.json", "w") as f:
    json.dump(results, f, indent=1)
print(f"saved results{run_tag}.json")

# ----------------------------------------------------------------------------
# 8. Plot
# ----------------------------------------------------------------------------
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"small": "#7A3E1D", "medium": "#A8663F", "large": "#1D4F7A"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in SIZE_NAMES:
        steps, losses = zip(*vanilla_curves[size]);    ax.plot(steps, losses, "--", color=colors[size], label=f"vanilla {size}")
        steps, losses = zip(*matryoshka_curves[size]); ax.plot(steps, losses, "-",  color=colors[size], label=f"matryoshka {size}")
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss (cross-entropy, chars)")
    ax.set_title(f"Three separate models (dashed) vs one nested model with three exits (solid)\n"
                 f"params: {vanilla_parameters/1e6:.2f}M vs {matryoshka_parameters/1e6:.2f}M "
                 f"({100*(1-matryoshka_parameters/vanilla_parameters):.0f}% fewer, parameter-matched per size)   "
                 f"alpha={args.alpha}  rescale={not args.no_rescale}", fontsize=10)
    ax.legend(); ax.grid(alpha=.3)
    out = f"curves{run_tag}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved {out}")
except ImportError:
    print("\n(matplotlib not installed — skipping the plot)")
