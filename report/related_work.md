# Related Work — Expansion Notes (W17)

This file structures the related-work section so the paper can cite
each line of research precisely and position our contribution against
it. The content is for the author's benefit, not the released bundle —
delete before publication if unwanted.

## 1. Unstructured magnitude-based pruning
- Han et al., *Learning both Weights and Connections for Efficient
  Neural Networks* (NeurIPS 2015). The original magnitude criterion our
  paper uses as a baseline.
- See et al., *Compression of Neural Machine Translation Models via
  Pruning* (2016). Evidence that magnitude pruning generalises across
  architectures but is sensitive to fine-tuning length.

**Our relation:** we treat magnitude as the strong baseline; show it
is not outperformed by Wanda on HAM10000 DeiT-Tiny / DeiT-Small.

## 2. Activation-aware pruning
- Sun et al., *A Simple and Effective Pruning Approach for Large
  Language Models* (Wanda, ICLR 2024). Score = |W| · ||a||.
- Xia et al., *Sheared LLaMA: Accelerating Language Model Pre-training
  via Structured Pruning* (2023). Discussion of outlier-channel
  protection.

**Our relation:** we implement Wanda for medical ViTs and show that
its outlier-protecting behaviour hurts rare dangerous-class
sensitivity. The Tier-1 activation-statistics evidence (kurtosis /
top-k concentration / outlier-ratio correlation with per-layer Wanda
damage) operationalises this hypothesis.

## 3. Gradient / Taylor-based pruning
- Molchanov et al., *Pruning Convolutional Neural Networks for Resource
  Efficient Inference* (ICLR 2017). First-order Taylor criterion.
- LeCun et al., *Optimal Brain Damage* (1989). Second-order foundations.

**Our relation:** Taylor is our second baseline. We show it matches
magnitude within seed noise on HAM10000 DeiT-Small.

## 4. Skewness-guided / distribution-aware pruning
- Paxton et al., *Skewness-Guided Pruning for Skin Lesion
  Classification* (MIDL 2022). Closest published method to our
  sensitivity-guided non-uniform allocation.

**Our relation:** implemented as `pruning.scoring.skewness_score`;
results in `baselines_paxton_xpruner.csv`. The paper should report
whether our method outperforms, matches, or underperforms theirs per
sparsity point.

## 5. Explainability-based pruning
- Yu et al., *X-Pruner: Explainable Pruning for Vision Transformers*
  (CVPR 2023). Channel-wise pruning via explainability maps.

**Our relation:** implemented as `pruning.scoring.xpruner_score`.

## 6. Second-order / OBS-family pruning
- Hassibi & Stork, *Optimal Brain Surgeon* (NeurIPS 1993).
- Frantar & Alistarh, *SparseGPT: Massive Language Models Can Be
  Accurately Pruned in One-Shot* (ICML 2023). Diagonal-OBS-like
  weight-error criterion used at scale.

**Our relation:** implemented as
`pruning.scoring.sparsegpt_pseudo_score`, using the Wanda-style
activation second moment as a cheap proxy for diag(H). Not a full
reproduction of SparseGPT; explicitly labelled as a diagonal
approximation in the text and the method name.

## 7. Structured and N:M sparsity
- Zhou et al., *Learning N:M Fine-grained Structured Sparse Neural
  Networks from Scratch* (ICLR 2021). Ampere-native 2:4 pattern.
- NVIDIA, *Accelerating Sparsity in the NVIDIA Ampere Architecture*
  (2020). Hardware support.

**Our relation:** `pruning.structured.NMPattern` + `e_structured_sparsity.py`
run 2:4 / 4:8 / 1:4 as hardware-addressable alternatives to our
unstructured masks (W8 follow-through).

## 8. Movement pruning
- Sanh et al., *Movement Pruning: Adaptive Sparsity by Fine-Tuning*
  (NeurIPS 2020). Prune during fine-tuning with learnable masks.

**Our relation:** `pruning.learnable_sparsity` is a close cousin at the
allocation level (learn per-layer keep ratios, not per-weight masks).
Movement pruning at the weight level is orthogonal to our contribution
and a natural combination for future work.

## 9. Knowledge distillation (pre-pruning preparation)
- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015).
- Touvron et al., *Training Data-Efficient Image Transformers* (DeiT,
  ICML 2021). Our teacher architecture family.

**Our relation:** experiments/e14_e16_distillation compares direct
supervised training vs. distilled students under Wanda pruning.

## 10. Federated / on-device training (out of scope)
- Mentioned in the paper introduction and the IoT assignment context
  but not methodologically central. Cited once; not expanded here.

## Contribution positioning

The paper's central contributions:

1. Negative result: Wanda underperforms magnitude / Taylor on HAM10000
   for DeiT-Tiny / DeiT-Small, with statistical support across 5 seeds
   and a concrete mechanistic explanation (distillation-driven
   activation concentration that Wanda over-protects).
2. Sensitivity-guided non-uniform allocation that rescues Wanda to the
   level of the best uniform criterion, validated against Paxton
   skewness and X-Pruner baselines.
3. A quantitative replacement for the cherry-picked attention figure,
   using ISIC segmentation masks on the full melanoma validation set
   (IoU + pointing-game accuracy), showing whether Wanda's "attention
   diffusion" is statistically meaningful.
