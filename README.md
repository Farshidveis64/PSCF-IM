# PSCF-IM: Path-Specific Counterfactual Fairness for Influence Maximization

A reproducible reference implementation of **PSCF-IM**, which brings
path-specific counterfactual fairness to influence maximization (IM). The
method separates the *fair* and *unfair* pathways by which a sensitive attribute
reaches activation and removes only the latter, instead of equalizing reach
across groups. It models influence through two latent mediators inferred from
cascades, defines a path-specific bias (**PS-Bias-IM**) via nested potential
outcomes, applies the minimal change to the diffusion model that drives it to
zero, and seeds under an edge-level fair Independent-Cascade model whose
`(1 − 1/e − ε)` greedy guarantee holds by construction.

---

## 1. Why two backends

| Backend | Module | Runs with | Role |
|---|---|---|---|
| **NumPy reference** | `pscf_im/models/estimator_numpy.py` | numpy + scikit-learn only | Dependency-free disentangler; powers the unit tests and every runnable experiment in this repo. |
| **PyTorch (production)** | `pscf_im/models/estimator_torch.py` | `+ torch` | Full GNN-VAE with an adversarial (GRL) disentangler and weak supervision; the path used for the paper's headline numbers. |

`backend='auto'` selects torch when it is importable and falls back to the NumPy
reference otherwise. **All results in this repository's `results/` were produced
by the NumPy reference backend**, so the on-axis disentanglement correlations
are moderate (≈0.6–0.9); the torch backend disentangles more sharply. The
qualitative findings — PSCF-IM drives PS-Bias-IM to near zero while staying
close to the unconstrained reach, and weak supervision is what makes the unfair
channel recoverable — hold under both backends.

---

## 2. Installation

```bash
# Option A: pip (core deps only; NumPy backend, all experiments, tests)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml && conda activate pscf-im

# Optional production backend + dev tools
pip install torch==2.3.1 pytest tqdm
```

Python 3.11+ is required.

---

## 3. Repository layout

```
pscf-im/
├── pscf_im/                    # importable package
│   ├── data/
│   │   ├── planted.py          # planted SBM + known SCM + oracle PS-bias
│   │   └── real.py             # SNAP edge-list loader / deterministic surrogate
│   ├── diffusion/
│   │   ├── ic.py               # Independent Cascade, live-edge worlds, reach
│   │   └── seeding.py          # CELF lazy greedy + RIS/IMM selection
│   ├── models/
│   │   ├── edge_model.py        # edge-level fair diffusion p^psf_uv (Eq. 10)
│   │   ├── estimator_numpy.py   # reference disentangler  (no torch)
│   │   ├── estimator_torch.py   # GNN-VAE + adversary      (torch)
│   │   └── grl.py               # gradient reversal layer
│   ├── metrics/fairness.py      # EQ, FR, recovery error, audit proxy PS*
│   ├── baselines/methods.py     # IMM / PIANO / EPIC / FIMM / UpLift / FAIM-RL / Total-Fairness
│   ├── evaluation.py            # frozen-oracle scoring + counterfactual PS-Bias-IM
│   └── pipeline.py              # infer → edit → fair-world seed             
├── configs/                     # default (demo) and certified (paper-scale) YAML
├── scripts/reproduce_all.sh     # end-to-end reproduction
├── main.py                      # unified entry point (dispatches all of the above)
├── requirements.txt / environment.yml
└── README.md
```

---

---

---

##  Method, end to end

**1. Planted SCM (`data/planted.py`).** A directed SBM with a known structural
causal model: a sensitive attribute `S`, a covariate `X`, a fair channel
`U_f` driven mostly by `X` with mild homophily on `S`, and an unfair channel
`U_b` driven by `S` and a continuous susceptibility latent. Because the SCM is
known, the path-specific bias of any node has a closed form — the change in its
outcome propensity when `S` is flipped *only along* `U_b`.

**2. Inference (`models/`).** The estimator recovers `(U_f, U_b)` from observed
signals. The reference backend implements the three load-bearing mechanisms of
the VAE without a network: a graph encoder (neighborhood smoothing), weak
supervision (the unfair channel is pinned by sparse flags `R_b`), and
adversarial disentanglement (the fair channel is fit on the residual after
orthogonally projecting out the recovered unfair direction — the linear analogue
of the GRL discriminator enforcing `U_f ⟂ U_b | S`).

**3. Minimal-change edit (`models/edge_model.py`).** The fair-world edge
probability `p^psf_uv = sigmoid(β₀ + ⟨U_f^u, W U_f^v⟩)` depends on the fair
channel only, so the unfair pathway is severed by construction. Because
`sigmoid` lands in the open unit interval, every probability is a valid
Bernoulli parameter with no side condition, so the induced process is a genuine
IC instance and its spread stays monotone submodular.

**4. Fair-world seeding (`diffusion/seeding.py`).** CELF lazy greedy (or RIS) on
a pre-sampled pool of live-edge worlds inherits the `(1 − 1/e − ε)` guarantee.

**5. Evaluation (`evaluation.py`).** All methods are scored under one *frozen
oracle* that is external to any seed-selection objective. PS-Bias-IM is the
**counterfactual flip**: the mean change in a policy's per-node activation when
`S` is flipped along the unfair channel, evaluated under the policy's own
committed diffusion with common random numbers. Fairness-unaware and statistical
baselines commit to the biased world (sensitive to the flip → large bias);
PSCF-IM commits to its inferred fair world (insensitive except through residual
disentanglement leakage → near zero).

---

## 7. Reproducibility guarantees

- A single entry point, `pscf_im.utils.seeding.set_global_seed`, fixes
  `random`, NumPy, `PYTHONHASHSEED`, and (when present) the torch CPU/CUDA
  generators and cuDNN flags.
- Live-edge worlds are pre-sampled once and shared across methods, so every
  comparison uses one coupling.
- Dependencies are pinned in `requirements.txt` / `environment.yml`.
- Each experiment dumps exact numeric values to CSV/JSON for downstream checks.

---

## 8. Real benchmark data

The four benchmarks are **not** redistributed. The loader uses the raw files
when present under `data/raw/` and otherwise builds a deterministic surrogate
with matching gross statistics, so everything runs without a download.

Public datasets can be fetched automatically (verified SNAP sources):

```bash
python main.py download --all          # email_eu, facebook, epinions
# or: python -m pscf_im.data.download --datasets email_eu
```

| name | edges file | labels | directed | source |
|---|---|---|---|---|
| `email_eu` | `email-Eu-core.txt` | `email-Eu-core-department-labels.txt` (42 depts) | yes | snap.stanford.edu/data/email-Eu-core.html |
| `facebook` | `facebook_combined.txt` | `facebook.labels` *(optional; else Louvain)* | no | snap.stanford.edu/data/ego-Facebook.html |
| `epinions` | `soc-Epinions1.txt` | *(Louvain proxy)* | yes | snap.stanford.edu/data/soc-Epinions1.html |
| `antelope_valley` | `antelope_valley.edges` | `antelope_valley.labels` | no | fair-IM literature (access-restricted) |

File formats and the Antelope Valley note are documented in
[`data/raw/README.md`](data/raw/README.md). Node IDs need not be contiguous —
the loader remaps them to `0..n-1` and aligns the label file through that map.
Only two benchmarks are demographic-labeled; the other two use a
feature/community proxy, so the audit proxy `PS*` is **suggestive, not
certified** (it conflates fair with unfair `S`-dependence). On real data the
full PSCF-IM pipeline infers the latent channels from observed cascades via the
torch backend; the runnable `run_realnet.py` reports the baselines and `PS*`.

---

## 9. Limitations (read before citing numbers)

- The committed `results/` come from the **NumPy reference** backend; use the
  torch backend and the paper-scale config for the headline disentanglement
  quality.
- The demo configuration uses small `n`, few worlds, and few seeds, so
  per-condition deltas carry visible Monte-Carlo noise; magnitudes sharpen at
  the paper scale.
- Declaring the fair/unfair partition does not remove the modeling burden — it
  relocates it onto the **provenance of the flags** `R_b`. Group-skewed flags
  can re-introduce bias (see `run_robustness.py`), which is why the paper
  recommends a mandatory subgroup audit of the flagging process.

---

## 10. License

MIT — see `LICENSE`.
