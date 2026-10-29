# Latent–Behavior Alignment in Language Models

**Can discrepancies between a model’s hidden representations and its observable behavior serve as a signal of hidden misalignment?**

This research project, conducted within the **HiveTrace research lab**, investigates **potential indicators of hidden misalignment** by comparing how language models judge harmful and safe statements with what can be decoded from their internal representations.

We use **Contrast-Consistent Search (CCS)** to extract layer-wise latent harmfulness scores and compare them with **behavioral harmfulness scores** from four Gemma 2 variants and a DeBERTa hate classifier. We then analyze the cases where these signals agree or diverge, and use **Polarity-Aware CCS (PA-CCS)** to study the consistency of latent representations across harmful/safe statement pairs.

## Research question

> **Can latent–behavioral disagreement help identify cases of potential hidden misalignment?**

Our working hypothesis is that a model’s observable judgment may not always reflect information that is decodable from its hidden states. Of particular interest are cases where **behavior disagrees with the dataset label while the latent probe agrees with it** (*behavior misaligned / latent aligned*). These are **candidates for further investigation**, not verified instances of deception or hidden misalignment.

Throughout the project, *aligned* and *misaligned* are **operational terms**: they refer to agreement or disagreement with the dataset’s harmful/safe labels. A probe’s ability to decode a signal does not demonstrate that this information is causally used to produce the model’s response.

## Key findings

- **Latent label agreement improves with depth.** In later layers, CCS scores generally separate harmful and safe statements more accurately; this pattern also appears under pair-wise cross-validation.
- **Potential hidden-misalignment signals can be isolated.** The *behavior misaligned / latent aligned* category persists even in late layers, while the opposite mismatch (*behavior aligned / latent misaligned*) generally decreases with depth. This motivates targeted audits of discrepant examples.
- **PA-CCS is sensitive to preprocessing.** Reference four-group centering and CCS-consistent two-center preprocessing yield substantially different ESA, PC and CI results, so conclusions about polarity consistency depend on the chosen convention.
- **Disagreement is suggestive, not conclusive.** Ambiguous annotations, prompt effects and probe artifacts may also produce mismatches. The current observational results **do not establish hidden misalignment** or that latent harmfulness information drives the model’s behavior.

The executed notebooks below contain figures, tables and examples. **Causal interventions in hidden states are a proposed next step, not part of this release.**

## Explore the results

The notebooks contain saved outputs and can be **viewed on GitHub without installing anything**:

| Model | Executed notebook |
| --- | --- |
| Gemma 2 2B | [01 — Gemma 2 2B](notebooks/01_gemma_2_2b.ipynb) |
| Gemma 2 2B Instruct | [02 — Gemma 2 2B IT](notebooks/02_gemma_2_2b_it.ipynb) |
| Gemma 2 9B | [03 — Gemma 2 9B](notebooks/03_gemma_2_9b.ipynb) |
| Gemma 2 9B Instruct | [04 — Gemma 2 9B IT](notebooks/04_gemma_2_9b_it.ipynb) |
| DeBERTa hate classifier | [05 — DeBERTa](notebooks/05_deberta_hate_tuned.ipynb) |
| Cross-model comparison | [06 — Depth aggregation](notebooks/06_aggregation.ipynb) |

The executed outputs were produced before this public-repository reorganization. Some displayed local paths therefore refer to the original working directory; the notebook code has been adapted to the new layout.

## Data and evaluation

The datasets come from [Sadiekh, Ericheva & Agarwal's polarity-probing project](https://github.com/SadSabrina/polarity-probing):

| Dataset | Statements | Harmful/safe pairs | Pair construction |
| --- | ---: | ---: | --- |
| `mixed` | 1,244 | 622 | Rephrasing and negation |
| `not` | 1,250 | 625 | Closely matched negations |

Each statement is also evaluated in **Yes** and **No** suffix forms for CCS representation extraction. These are different from the semantic harmful/safe pairs.

**Behavioral score.** Gemma models receive a fixed harmfulness question, and their Yes/No logits yield a forced-choice probability of harmfulness. DeBERTa uses its hate/non-hate classification logits. Behavioral decision thresholds are selected using training labels only.

**Latent score.** A linear CCS probe is fitted to hidden representations from the Yes/No suffix pair without a supervised classification-loss term. Its sign is oriented using **training labels**. The label-oriented score is then evaluated on held-out statements.

**Splits.** We report both the original row-wise 80/20 split and **five-fold pair-wise cross-validation**, which keeps both statements of each semantic harmful/safe pair in the same fold. The pair-wise protocol is the stronger held-out-pair check; the two protocols do not use identical test populations. In the row-wise PA-CCS evaluation, the semantic counterpart of a selected statement can come from the other split (`counterpart_outside_split` is recorded).

**PA-CCS metrics.** ESA measures sign-invariant separation of harmful/safe classes; Polar Consistency (PC) and Contradiction Index (CI) characterize behavior under polarity reversal. We keep two variants separate:

- `reference`: independent centering of the four evaluation groups (safe/harmful × Yes/No).
- `consistent`: the same **two train-derived Yes/No centers** used by the CCS probe, applied without refitting on evaluation groups.

The `small` aggregate includes both Gemma 2B variants **and DeBERTa**; model size is therefore not isolated from architecture in that comparison.

## Repository layout

```text
README.md
requirements.txt
notebooks/               # 5 model notebooks + depth aggregation
modules/
    artifacts.py          # Dataset and cached-input validation
    ccs.py                # Bundled reference CCS implementation
    experiments.py        # Splits, probes, scores, PA metrics, aggregation
    plots.py              # Figures and geometry visualizations
```

Only the **code and executed notebooks** are published here. Original CSVs, extracted hidden states, logits, probe caches and standalone result tables are not included. The notebook outputs allow inspection, but reproducing the computations requires those external artifacts.

## Limitations and next steps

The central question — **whether latent–behavioral discrepancies reliably signal hidden misalignment** — remains open. This study identifies and characterizes disagreements, but cannot distinguish hidden misalignment from every alternative explanation.

- **Dataset validity:** Some harmful/safe labels are ambiguous and need manual audit.
- **Probe validity:** CCS may extract correlates or artifacts rather than a robust harmfulness representation; additional controls are needed.
- **Behavioral protocol:** The behavioral scores reflect a fixed prompt or classifier output, not all possible responses or free-form generation.
- **Causality:** Decodability does not imply causal use. Interventions along the CCS direction and checks for nonspecific output degradation would be needed to test whether the decoded signal influences behavior.
- **Generalization:** Results depend on dataset construction, model family and preprocessing choices; replication on new prompts and data remains necessary.

**Next steps:** label audit → control experiments → validation across prompts and datasets → causal interventions in hidden states.

## Attribution

This project builds on the datasets and PA-CCS methodology of **Sadiekh, Ericheva & Agarwal (2026)** and the original **CCS** framework of **Burns et al. (2023)**. The `modules/ccs.py` implementation is reproduced from the linked polarity-probing source and should retain its original authorship attribution.

- Sadiekh, S., Ericheva, E., & Agarwal, C. (2026). [*Polarity-Aware Probing for Quantifying Latent Alignment in Language Models*](https://doi.org/10.1609/aaai.v40i44.41126). AAAI-26.
- Burns, C., et al. (2023). [*Discovering Latent Knowledge in Language Models Without Supervision*](https://arxiv.org/abs/2212.03827). ICLR 2023.
- [Original code and datasets: polarity-probing](https://github.com/SadSabrina/polarity-probing).
