**TinyML / Efficient Deep Learning**

**Final Project Options — Enhanced Edition**

*Two fully-detailed proposals with maximum course-topic coverage*

Team: 2 ML majors  |  Timeline: 2–3 weeks  |  Compute: MBZUAI cluster

Goal: Best possible grade with 3–4/5 novelty  |  Tooling: PyTorch, timm, Claude Code

# **Executive summary**

This document presents two enhanced project proposals. Both have been upgraded from earlier versions to cover three out of five major course concepts (pruning, quantization, and knowledge distillation), grounded in real-world deployment problems, and designed for maximum grade potential within our 2–3 week timeline.

*Does Compression Forget Cancer? — Pruning \+ quantization \+ distillation of medical ViTs for edge dermatology, with diagnostic safety analysis and sensitivity-guided layer-wise sparsity allocation.*

# **Does Compression Forget Cancer?**

**Full title:** *Efficient Medical Vision Transformers for Edge Dermatology: Pruning, Quantization, and Distillation with Diagnostic Safety Analysis.*

*RECOMMENDED. Combines three course techniques with a powerful real-world framing and genuine novelty.*

## **A.1 The real-world problem**

Skin cancer kills over 60,000 people globally each year. AI-powered dermatoscopes can match dermatologist-level accuracy, but the models are too large for handheld devices in clinics without cloud connectivity. To deploy on edge, we must compress. But medical data is highly imbalanced: roughly 90% of lesions are benign, 10% are dangerous (melanoma, basal cell carcinoma). When we compress an AI, does it forget the rare, deadly diseases first? And which combination of compression techniques best preserves diagnostic safety?

## **A.2 What we will do**

We apply three compression techniques from the course — pruning, quantization, and knowledge distillation — to Vision Transformers fine-tuned on the HAM10000 skin lesion dataset. Our study has four experimental pillars:

1. Pruning criterion comparison: Magnitude, Wanda (activation-aware), Taylor one-shot, and Random across five sparsity levels, evaluated with per-class clinical metrics (sensitivity for melanoma and other dangerous classes).

2. Sensitivity-guided layer-wise sparsity allocation (HALM-inspired): Use the per-layer sensitivity analysis to assign non-uniform sparsity across the ViT — more aggressive pruning for robust layers, gentler for diagnostically critical ones. Compare against uniform sparsity at matched overall compression.

3. Post-pruning quantization: Apply INT8 PTQ on top of the best pruned model to test whether pruning and quantization stack cleanly, or whether the combination breaks diagnostic safety.

4. Distillation before pruning: Use unpruned DeiT-Small as teacher to distill into DeiT-Tiny before pruning. Compare DeiT-Tiny pruned directly vs distilled-then-pruned DeiT-Tiny. Tests whether KD gives the student better features that survive compression.

## **A.3 Why this is a great project for us**

* Covers 3 out of 5 course concepts (pruning, quantization, KD) in one coherent project with a shared evaluation framework.

* Real-world motivation that graders immediately respect: medical AI safety under compression.

* Inference-only core for the pruning/quantization legs. Training is limited to fine-tuning (\~30 min) and one KD run (\~30 min).

* Clinical metrics (per-class sensitivity) elevate a standard benchmark into a meaningful contribution.

* The question “Does activation-aware pruning protect melanoma sensitivity better than magnitude?” is genuinely unanswered.

* The HALM-inspired non-uniform allocation adds a method contribution, not just a comparison.

* Failure-tolerant at every level: null results in any pillar are findings, not failures.

* Presentation is visually compelling: dermoscopic images, per-class sensitivity crash charts, per-layer heatmaps.

## **A.4 Novelty angle and assessment**

The novelty is layered — each layer is independently defensible:

5. Cross-domain transfer: Wanda (designed for LLMs) applied to medical ViTs. Do medical image patches have activation outliers like text tokens?

6. Diagnostic safety under compression: Shifting from overall accuracy to class-specific clinical metrics in the context of pruning. The question “which criterion forgets cancer first?” is a fresh framing.

7. Sensitivity-guided non-uniform sparsity (HALM-inspired): Using per-layer sensitivity scores to allocate sparsity non-uniformly. This is a method contribution, not just a comparison.

8. Compression stacking: Does pruning \+ quantization degrade clinical safety more than either alone? Interaction effects between techniques are under-studied for medical models.

9. Distillation as pre-treatment: Does KD before pruning produce features that are more robust to subsequent compression? The order-of-operations question applied to medical ViTs.

**Novelty score estimate: 3.5–4 / 5\.**

**Is the novelty real?** Yes. Individual components exist separately, but the specific combination — Wanda \+ HALM-style allocation \+ stacked quant \+ KD pre-treatment, all evaluated with clinical diagnostic safety metrics on medical ViTs — is new and defensible.

## **A.5 Quick reference**

| Dataset | HAM10000 (10,015 dermoscopic images, 7 classes: melanoma, melanocytic nevus, BCC, actinic keratosis, benign keratosis, dermatofibroma, vascular lesion). Freely available. \~3 GB. |
| :---- | :---- |
| **Models** | DeiT-Tiny (\~5M params) and DeiT-Small (\~22M). Pretrained on ImageNet via timm, fine-tuned on HAM10000 with weighted cross-entropy. |
| **Course concepts** | Pruning (4 criteria, uniform \+ non-uniform allocation), Quantization (INT8 PTQ stacked on pruning), Knowledge Distillation (DeiT-Small → DeiT-Tiny before pruning). |
| **Key metrics** | Per-class sensitivity (recall), balanced accuracy, overall accuracy, F1 per class, model size (KB), inference latency (ms). |
| **Compute estimate** | \~12–18 GPU-hours total. |
| **Total experiments** | Fine-tune 2 models \+ 1 KD run (\~2 hrs). Pruning matrix: 4 criteria × 5 sparsities × 2 models \= 40 evals. Per-layer analysis: \~8 evals. Non-uniform allocation: \~6 evals. Stacked quant: \~4 evals. KD+prune comparison: \~8 evals. Total: \~70 experiments. |

## **A.6 Full experimental plan**

### **Pillar 1 — Pruning criterion comparison (core)**

* E1. Fine-tune DeiT-Tiny and DeiT-Small on HAM10000 with weighted cross-entropy loss (to handle class imbalance), standard augmentation (random crop, flip, color jitter), stratified 80/20 split. \~15–30 min per model.

* E2. Baseline evaluation: per-class sensitivity, balanced accuracy, confusion matrix for both models before any compression.

* E3. Calibration pass: 128 HAM10000 training images forwarded through each model, capturing input activations per linear layer via register\_forward\_hook.

* E4. Pruning matrix: for each criterion (magnitude, Wanda, Taylor, random) at each sparsity (20%, 40%, 50%, 60%, 70%), apply one-shot unstructured pruning and evaluate. Record overall accuracy, balanced accuracy, and per-class sensitivity. 40 evaluations.

* E5. Per-layer-type breakdown at 50% sparsity: prune only QKV / only MLP / only attention output / only patch embedding, for Wanda and magnitude on DeiT-Small. 8 evaluations.

* E6. Diagnostic safety analysis: for each (criterion, sparsity), compute melanoma sensitivity drop relative to unpruned baseline. Build the HEADLINE figure: melanoma sensitivity vs sparsity, one line per criterion.

### **Pillar 2 — Sensitivity-guided non-uniform allocation (HALM-inspired)**

* E7. Using the per-layer sensitivity data from E5, compute a sensitivity score per layer (accuracy drop when that layer is pruned at 50%).

* E8. Create a non-uniform sparsity policy: layers with low sensitivity get 70% sparsity, medium-sensitivity layers get 50%, high-sensitivity layers get 30%. The overall average sparsity should match a uniform 50% baseline.

* E9. Apply the non-uniform policy with Wanda and magnitude scoring. Compare against uniform 50% for both criteria. 4 evaluations.

* E10. Report per-class sensitivity under non-uniform vs uniform allocation. The question: does smart allocation protect melanoma sensitivity better than uniform pruning?

### **Pillar 3 — Post-pruning quantization (stacking test)**

* E11. Take the best Wanda-pruned and best magnitude-pruned models (at 50% sparsity). Apply INT8 PTQ using torch.ao.quantization with a small calibration set (128 images).

* E12. Evaluate the pruned+quantized models with the same per-class metrics. Compare: dense → pruned-only → pruned+quantized. 4 evaluations.

* E13. Report the interaction: does stacking quantization on top of pruning cause a further melanoma sensitivity drop, or does INT8 barely affect the already-pruned model?

### **Pillar 4 — Distillation before pruning (KD pre-treatment)**

* E14. Knowledge distillation: use unpruned DeiT-Small as teacher, train DeiT-Tiny as student with soft labels (temperature=4, alpha=0.7). \~30 min training.

* E15. Compare three DeiT-Tiny variants: (A) fine-tuned directly on HAM10000 (already done in E1), (B) distilled from DeiT-Small (E14), (C) pretrained ImageNet only (no HAM10000 fine-tuning, as a lower-bound control).

* E16. Prune all three DeiT-Tiny variants with Wanda at 50% sparsity. Compare per-class sensitivity. The question: does KD produce a DeiT-Tiny whose features survive pruning better? 6 evaluations.

### **Optional extensions — if time permits**

* E17. Short finetuning recovery (1–2 epochs) after pruning for the best configurations. Shows whether the criterion gap survives recovery.

* E18. Attention-map visualization on melanoma images: dense vs Wanda-pruned vs magnitude-pruned. Tests whether Wanda preserves attention on the lesion area.

* E19. Calibration-set-size ablation for Wanda (16 / 128 / 512 images).

* E20. Add MobileNetV2 as a CNN baseline to test architecture generality.

## **A.7 Evaluation metrics**

* Per-class sensitivity (recall) — the primary metric, especially for melanoma, BCC, and actinic keratosis.

* Balanced accuracy (macro-average of per-class recall) — the fairness-aware aggregate metric.

* Overall accuracy — reported but explicitly shown to be misleading on imbalanced data.

* Per-class F1 score.

* Model size in KB (original, pruned, pruned+quantized, distilled, distilled+pruned).

* CPU inference time in ms.

* Compression ratio relative to the dense baseline.

## **A.8 Expected figures and tables**

* Figure 1 (HEADLINE): Melanoma sensitivity vs sparsity, one line per pruning criterion. Shows when each criterion starts forgetting cancer.

* Figure 2: Balanced accuracy vs sparsity for all four criteria, per model.

* Figure 3: Per-layer-type sensitivity drop at 50% sparsity (grouped bars). Answers: which ViT components encode diagnostic features?

* Figure 4: Non-uniform vs uniform sparsity comparison at matched overall sparsity. Shows whether HALM-style allocation protects clinical safety.

* Figure 5: Compression stacking — dense vs pruned-only vs pruned+quantized, per-class sensitivity bars.

* Figure 6: KD pre-treatment — direct fine-tune vs distilled DeiT-Tiny, per-class sensitivity after pruning at 50%.

* Figure 7 (optional): Attention-map visualization on melanoma images.

* Table 1: Master results matrix — technique × sparsity × {overall acc, balanced acc, melanoma recall, BCC recall, size KB}.

* Table 2: Per-class sensitivity at 50% sparsity for all criteria (the diagnostic safety table).

* Table 3: Compression pipeline comparison — pruning-only vs pruning+quant vs KD+pruning vs KD+pruning+quant.

## **A.9 Three-week execution plan**

| Week | Days | Focus | Deliverable |
| :---- | :---- | :---- | :---- |
| **1** | **1–2** | Download HAM10000, set up stratified splits. Fine-tune DeiT-Tiny and DeiT-Small with weighted CE. Verify baseline per-class metrics. | *2 fine-tuned models, baseline metrics.* |
| **1** | **3** | Implement calibration pass (hooks on INPUT activations), 4 scoring functions, masking utility, per-class evaluation. Sanity check: 0% sparsity \= baseline. | *Working pruning \+ evaluation pipeline.* |
| **1** | **4–5** | Run Pillar 1: pruning matrix (40 evaluations) \+ per-layer-type breakdown (8 evals). Build headline melanoma sensitivity figure. | *Figures 1, 2, 3 drafted.* |
| **1** | **6** | Run Pillar 2: sensitivity-guided non-uniform allocation (4–6 evals). Compare against uniform at matched sparsity. | *Figure 4 drafted.* |
| **1** | **7** | Buffer \+ sanity checks \+ first plotting pass. | *Clean Week 1 results.* |
| **2** | **8** | Run Pillar 3: INT8 PTQ on best pruned models (4 evals). Build stacking figure. | *Figure 5 drafted.* |
| **2** | **9–10** | Run Pillar 4: KD from DeiT-Small to DeiT-Tiny (\~30 min). Prune all three DeiT-Tiny variants at 50% with Wanda (6 evals). Build KD pre-treatment figure. | *Figure 6 drafted.* |
| **2** | **11–12** | Finalize all figures with consistent styling. Build presentation slides. Rehearse. | *Slide deck ready.* |
| **2** | **13–14** | Present. Begin report introduction and method sections. | *Presentation delivered.* |
| **3** | **15–16** | Optional extensions: finetuning recovery, attention-map viz, calibration ablation. | *Extension data.* |
| **3** | **17–18** | Full report writing: intro, related work, method, experiments, results, analysis. | *Report draft.* |
| **3** | **19–21** | Polish, cross-read, finalize figures and tables, submit. | *Final report.* |

## **A.10 Division of work**

* **Person A (Code and infrastructure):** Implements scoring functions, calibration hooks, masking utility, per-layer-type filters, non-uniform allocation logic, INT8 PTQ integration. Fine-tunes models. Writes method section of report.

* **Person B (Experiments, KD, and analysis):** Runs the pruning matrix, implements KD training loop (DeiT-Small → DeiT-Tiny), manages logging, builds all per-class sensitivity plots, leads the diagnostic safety analysis. Writes results and analysis sections.

* **Shared:** Data preprocessing, presentation slides, rehearsal, report intro/related work/conclusion, final polish.

## **A.11 Risks and mitigations**

* R1. HAM10000 class imbalance (melanocytic nevus dominates). MITIGATION: weighted cross-entropy during fine-tuning, report balanced accuracy, use stratified splits.

* R2. ViT on HAM10000 may underperform published CNN baselines in absolute accuracy. MITIGATION: expected and acceptable — our goal is the compression comparison, not SOTA classification. Report honestly.

* R3. Wanda hook bug (capturing outputs instead of inputs). MITIGATION: explicit shape sanity check before running the matrix.

* R4. Non-uniform allocation may not beat uniform (layers may have similar sensitivity). MITIGATION: this is itself a finding — report it honestly. Shows that ViT layers in medical context are uniformly important.

* R5. INT8 PTQ may barely affect accuracy (INT8 is gentle). MITIGATION: if so, report that quantization stacks safely on top of pruning for medical ViTs. Try INT4 via bitsandbytes if INT8 is too gentle.

* R6. KD run may not converge well in limited epochs. MITIGATION: use standard hyperparameters (T=4, alpha=0.7), AdamW, cosine LR. If student underperforms, extend to 10 more epochs.

* R7. Melanoma sensitivity may be low even at 0% sparsity due to imbalance. MITIGATION: report both absolute sensitivity and delta (drop from baseline).

## **A.12 Presentation and report focus**

The presentation opens with the real-world problem: “When we compress a medical AI to fit on a phone, which method best protects the ability to detect deadly diseases?” It walks through the four pillars: pruning criteria comparison (Figure 1), smart layer allocation (Figure 4), stacking quantization (Figure 5), and KD pre-treatment (Figure 6). The narrative arc builds from simple observation (criterion comparison) to proposed method (non-uniform allocation) to deployment pipeline (stacking). Each figure has a clinical interpretation.

The report adds the full results matrix, ablations, per-layer heatmaps, and a discussion section addressing: (a) whether activation outliers from LLMs exist in medical image patches, (b) which ViT components encode diagnostically critical features, (c) recommendations for practitioners compressing medical ViTs.

## **A.13 Pitch text for the professor**

*Project Title: Does Compression Forget Cancer? Pruning, Quantization, and Distillation of Medical Vision Transformers with Diagnostic Safety Analysis. Motivation: Deploying skin cancer screening AI on edge devices requires aggressive compression, but medical data is highly imbalanced. Standard compression metrics (overall accuracy) mask dangerous drops in rare-disease detection. Methodology: We fine-tune DeiT-Tiny and DeiT-Small on HAM10000 and evaluate three compression techniques: (i) one-shot pruning with four criteria including Wanda (activation-aware, from the LLM literature), with sensitivity-guided layer-wise sparsity allocation; (ii) INT8 post-training quantization stacked on pruning; (iii) knowledge distillation as pre-treatment before pruning. All evaluated with per-class diagnostic sensitivity, focusing on melanoma and clinically critical classes. Expected contribution: A concrete answer to which compression pipeline best preserves diagnostic safety for medical ViTs, with a proposed sensitivity-guided allocation strategy and deployment recommendations.*

## **A.14 Classification**

**Strong, feasible, covers 3/5 course concepts, genuine 3.5–4/5 novelty, compelling real-world framing. RECOMMENDED.**

