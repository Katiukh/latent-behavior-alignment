# Fixed-threshold latent–behavior mismatch

Inputs are existing scores.csv files only. Source hashes and run settings are in
manifest.json. No hidden states, probe training or behavioral logits are computed.

For true_label=1, alignment equals the corresponding hate probability. For
true_label=0, alignment equals 1 minus that probability. Latent probability uses
the already oriented latent_score. Gemma behavioral_probability_yes is copied to
behavioral_probability_hate: the checkpoint2 prompt asks whether a statement is
harmful. Original columns (including chat probabilities/responses) are preserved;
chat scores do not replace the main behavioral probability.

Both aligned flags use alignment >= 0.5. A value above 0.5 supports the true class;
a value below supports the other class. Exactly 0.5 is neutral and is assigned to
aligned by the specified inclusive boundary. This threshold is fixed, not fitted.

| behavior_aligned | latent_aligned | case_type |
| --- | --- | --- |
| True | True | both_aligned |
| True | False | aligned_output_misaligned_latent |
| False | True | misaligned_output_aligned_latent |
| False | False | both_misaligned |

signed_alignment_gap = latent_alignment - behavioral_alignment. Positive values
mean the latent score agrees more strongly with the true label; negative values
mean the behavioral score does. abs_alignment_gap is its absolute magnitude.

Statistics are computed separately per dataset/model/layer across all existing
splits; split remains in the object tables. Each object may recur across layers.
All four case types are included: empty cases have n=0, fraction=0 and blank
descriptive statistics. Summary medians only describe already assigned cases.
Each fraction uses the total number of objects in that layer as denominator.

mismatch_fractions.csv contains all four fractions in one row per model/layer.
top_mismatch_cases.csv includes up to 20 cases per mismatch category per layer
(configurable with --top-n), ranked by decreasing abs_alignment_gap. Ties are
ordered by sample_idx. The other two categories do not enter these rankings.

plots/layer_XX.png shows behavioral_alignment versus latent_alignment with fixed
[0,1] axes, the horizontal/vertical 0.5 boundaries, a y=x diagonal and colors by
case_type. No clustering is performed.

Reproduce: python notebooks/checkpoint2/analyze_mismatch_cases.py
Requires numpy, pandas and matplotlib. Dataset-wide tables live in this directory;
model-specific tables and plots live in each model directory. Combined summaries
for both datasets are results/checkpoint2/mismatch_summary_all.csv and
results/checkpoint2/mismatch_fractions_all.csv.
