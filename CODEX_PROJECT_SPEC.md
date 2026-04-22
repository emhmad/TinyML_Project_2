# Code Documentation — "Does Compression Forget Cancer?"
# Efficient Medical Vision Transformers for Edge Dermatology
# Feed this entire file to Codex / Claude Code as the project specification.

---

## 1. PROJECT OVERVIEW

**Goal:** Compare pruning criteria (Magnitude, Wanda, Taylor, Random) on Vision Transformers fine-tuned for skin lesion classification (HAM10000). Evaluate with per-class clinical metrics (especially melanoma sensitivity). Additionally test sensitivity-guided non-uniform sparsity allocation, INT8 post-training quantization stacked on pruning, and knowledge distillation as pre-treatment before pruning.

**Key research question:** When we compress a medical ViT, does it forget rare deadly diseases (melanoma) first? Does activation-aware pruning (Wanda) protect diagnostic safety better than magnitude pruning?

**Framework:** PyTorch
**Models:** DeiT-Tiny, DeiT-Small (from `timm`)
**Dataset:** HAM10000 (10,015 dermoscopic images, 7 classes)
**Hardware:** NVIDIA A100 GPU (MBZUAI cluster)

---

## 2. REPOSITORY STRUCTURE

```
project/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml              # All hyperparameters in one place
├── data/
│   ├── download_ham10000.py      # Download and extract HAM10000
│   └── dataset.py                # HAM10000Dataset class + transforms + splits
├── models/
│   ├── load_models.py            # Load DeiT-Tiny/Small from timm, adapt head
│   └── distillation.py           # KD training loop (Pillar 4)
├── pruning/
│   ├── scoring.py                # 4 scoring functions: magnitude, wanda, taylor, random
│   ├── hooks.py                  # Activation hooks for Wanda and Taylor
│   ├── masking.py                # Apply masks, compute sparsity stats
│   ├── layer_groups.py           # Identify and group ViT layer types (QKV, MLP, attn_out, patch_embed)
│   └── nonuniform.py             # HALM-inspired sensitivity-guided allocation (Pillar 2)
├── quantization/
│   └── ptq.py                    # INT8 post-training quantization (Pillar 3)
├── evaluation/
│   ├── metrics.py                # Per-class sensitivity, balanced acc, F1, confusion matrix
│   ├── latency.py                # CPU/GPU inference latency measurement
│   └── model_size.py             # Model size in KB (dense, sparse, quantized)
├── experiments/
│   ├── e1_finetune.py            # Fine-tune DeiT-Tiny and DeiT-Small on HAM10000
│   ├── e2_baseline_eval.py       # Baseline per-class metrics before compression
│   ├── e3_calibration.py         # Calibration forward pass to collect activations
│   ├── e4_pruning_matrix.py      # Run 4 criteria × 5 sparsities × 2 models
│   ├── e5_perlayer_breakdown.py  # Per-layer-type pruning analysis
│   ├── e6_diagnostic_safety.py   # Build headline melanoma sensitivity figure
│   ├── e7_e10_nonuniform.py      # Pillar 2: non-uniform allocation experiments
│   ├── e11_e13_quantization.py   # Pillar 3: stacked PTQ experiments
│   ├── e14_e16_distillation.py   # Pillar 4: KD pre-treatment experiments
│   └── run_all.py                # Master experiment runner
├── plotting/
│   ├── fig1_melanoma_sensitivity.py
│   ├── fig2_balanced_accuracy.py
│   ├── fig3_perlayer_bars.py
│   ├── fig4_nonuniform_vs_uniform.py
│   ├── fig5_stacking.py
│   ├── fig6_kd_pretreatment.py
│   └── style.py                  # Shared plot styling (fonts, colors, sizes)
├── results/                      # Auto-populated by experiments
│   ├── logs/                     # CSV logs per experiment
│   └── checkpoints/              # Saved models and masks
└── notebooks/
    └── analysis.ipynb            # Interactive analysis and figure review
```

---

## 3. DEPENDENCIES

```
# requirements.txt
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.12
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.65.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 4. CONFIGURATION

```yaml
# configs/default.yaml

# --- Dataset ---
dataset:
  name: HAM10000
  root: ./data/ham10000/
  image_size: 224
  train_split: 0.8          # stratified
  seed: 42
  num_workers: 4

# --- Models ---
models:
  teacher: deit_small_patch16_224    # timm model name
  student: deit_tiny_patch16_224     # timm model name
  num_classes: 7
  pretrained: true                   # ImageNet pretrained

# --- Fine-tuning ---
finetune:
  epochs: 20
  batch_size: 64
  lr: 1.0e-4
  weight_decay: 0.05
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 2
  use_weighted_loss: true            # weighted CE for class imbalance

# --- Augmentation ---
augmentation:
  random_crop: 224
  horizontal_flip: true
  vertical_flip: true
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

# --- Pruning ---
pruning:
  criteria: [magnitude, wanda, taylor, random]
  sparsities: [0.2, 0.4, 0.5, 0.6, 0.7]
  calibration_size: 128              # images for Wanda/Taylor calibration
  exclude_layers:                    # do NOT prune these
    - head                           # classifier head
    - cls_token
    - pos_embed
    - patch_embed.proj               # patch embedding projection (optional, discuss in report)

# --- Non-uniform allocation (Pillar 2) ---
nonuniform:
  target_avg_sparsity: 0.5
  bins:
    low_sensitivity: 0.7             # layers that tolerate pruning well
    medium_sensitivity: 0.5
    high_sensitivity: 0.3            # layers critical for diagnostics

# --- Quantization (Pillar 3) ---
quantization:
  backend: fbgemm                    # or qnnpack for ARM
  calibration_size: 128
  dtype: qint8

# --- Knowledge Distillation (Pillar 4) ---
distillation:
  temperature: 4.0
  alpha: 0.7                         # weight for soft labels (1-alpha for hard labels)
  epochs: 20
  batch_size: 64
  lr: 1.0e-4
  optimizer: adamw
  scheduler: cosine

# --- Evaluation ---
evaluation:
  batch_size: 128
  latency_warmup_runs: 10
  latency_timed_runs: 100

# --- Logging ---
logging:
  results_dir: ./results/logs/
  checkpoints_dir: ./results/checkpoints/
```

---

## 5. DATA PIPELINE

### 5.1 `data/download_ham10000.py`

```
PURPOSE: Download HAM10000 dataset, extract, and organize into a standard directory.

STEPS:
1. Download from Harvard Dataverse or Kaggle (HAM10000).
   - Images: ~10,015 dermoscopic images (.jpg)
   - Metadata: HAM10000_metadata.csv with columns:
     lesion_id, image_id, dx (diagnosis label), dx_type, age, sex, localization
2. The 7 classes in the `dx` column:
   - 'mel'   → Melanoma (DANGEROUS - the primary focus class)
   - 'nv'    → Melanocytic nevus (benign, ~67% of data - MAJORITY)
   - 'bcc'   → Basal cell carcinoma (DANGEROUS)
   - 'akiec' → Actinic keratosis (DANGEROUS)
   - 'bkl'   → Benign keratosis
   - 'df'    → Dermatofibroma
   - 'vasc'  → Vascular lesion
3. Create label encoding: {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
4. Save processed metadata CSV with columns: image_path, label_idx, label_name

OUTPUT: ./data/ham10000/ directory with images and processed metadata CSV.
```

### 5.2 `data/dataset.py`

```python
"""
HAM10000 Dataset class.

IMPORTANT REQUIREMENTS:
- Stratified train/val split preserving class proportions.
- Compute class weights for weighted cross-entropy (inverse frequency).
- Return: (image_tensor, label_idx) where image_tensor is [3, 224, 224].
"""

class HAM10000Dataset(torch.utils.data.Dataset):
    """
    Args:
        metadata_csv: path to processed CSV
        image_dir: path to image directory
        transform: torchvision transforms
        indices: subset indices for train/val split
    
    Returns:
        image: Tensor [3, 224, 224] normalized with ImageNet stats
        label: int in [0, 6]
    """

def get_train_val_splits(metadata_csv, train_ratio=0.8, seed=42):
    """
    Stratified split using sklearn.model_selection.StratifiedShuffleSplit.
    
    Returns:
        train_indices: list of int
        val_indices: list of int
    """

def compute_class_weights(metadata_csv, train_indices):
    """
    Compute inverse-frequency weights for weighted cross-entropy.
    
    Returns:
        weights: Tensor of shape [7], normalized so sum = 7.
        Example: melanocytic nevus (majority) gets weight ~0.3,
                 melanoma gets weight ~3.0.
    """

def get_transforms(split='train'):
    """
    Returns torchvision.transforms.Compose.
    
    Train: Resize(256) → RandomCrop(224) → RandomHorizontalFlip → 
           RandomVerticalFlip → ColorJitter → ToTensor → Normalize
    Val:   Resize(256) → CenterCrop(224) → ToTensor → Normalize
    """
```

---

## 6. MODEL LOADING

### 6.1 `models/load_models.py`

```python
"""
Load pretrained DeiT models from timm and adapt the classification head.

CRITICAL: The default timm DeiT has a 1000-class ImageNet head.
We must replace it with a 7-class head for HAM10000.
"""

def load_deit_model(model_name: str, num_classes: int = 7, pretrained: bool = True):
    """
    Args:
        model_name: one of 'deit_tiny_patch16_224', 'deit_small_patch16_224'
        num_classes: 7 for HAM10000
        pretrained: load ImageNet pretrained weights
    
    Returns:
        model: nn.Module with replaced head
    
    Implementation:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        # timm handles head replacement when num_classes != 1000
    """

def get_linear_layer_names(model):
    """
    Return a list of (name, module) for all nn.Linear layers in the model.
    Used by pruning code to identify which layers to prune.
    
    IMPORTANT: Exclude the classification head ('head') and 
    any layers in the exclude list from configs.
    
    Returns:
        list of tuples: [(name, nn.Linear), ...]
        
    Example output for DeiT-Tiny:
        [('blocks.0.attn.qkv', Linear(192, 576)),
         ('blocks.0.attn.proj', Linear(192, 192)),
         ('blocks.0.mlp.fc1', Linear(192, 768)),
         ('blocks.0.mlp.fc2', Linear(768, 192)),
         ('blocks.1.attn.qkv', Linear(192, 576)),
         ...]
    """

def classify_layer_type(layer_name: str) -> str:
    """
    Classify a layer name into one of 4 ViT component types.
    
    Rules:
        'attn.qkv'  → 'qkv'        (Q, K, V projection)
        'attn.proj'  → 'attn_out'   (attention output projection)
        'mlp.fc1'    → 'mlp'        (MLP first layer)
        'mlp.fc2'    → 'mlp'        (MLP second layer)
        'patch_embed' → 'patch_embed'
        'head'       → 'head'       (excluded from pruning)
    
    Returns:
        str: one of ['qkv', 'attn_out', 'mlp', 'patch_embed', 'head']
    """
```

---

## 7. PRUNING MODULE

### 7.1 `pruning/hooks.py`

```python
"""
Activation hooks for Wanda and Taylor scoring.

CRITICAL BUG TO AVOID: 
- Wanda needs INPUT activations to each Linear layer, NOT outputs.
- The hook signature is: hook(module, input, output)
- We want input[0], not output.
"""

class ActivationCollector:
    """
    Registers forward hooks on all target Linear layers.
    Collects the L2 norm of input activations per feature.
    
    Usage:
        collector = ActivationCollector(model, target_layers)
        collector.register_hooks()
        
        # Forward calibration batch
        with torch.no_grad():
            for images, _ in calibration_loader:
                model(images.to(device))
        
        activation_norms = collector.get_activation_norms()
        collector.remove_hooks()
    
    Methods:
        register_hooks() → None
            Registers register_forward_hook on each target layer.
            
        get_activation_norms() → dict[str, Tensor]
            Returns {layer_name: activation_norm_tensor} where
            activation_norm_tensor has shape [in_features].
            The norm is the L2 norm of input activations averaged
            across all calibration samples and all tokens/spatial positions.
            
            For a Linear layer with input shape [batch, seq_len, in_features]:
                norm = mean over (batch, seq_len) of ||x||_2 per feature
            
        remove_hooks() → None
            Removes all registered hooks.
    
    SANITY CHECK:
        After collecting, verify that activation_norms[layer_name].shape[0] 
        == layer.in_features for each layer. If it equals out_features, 
        you captured the OUTPUT, not input. This is the #1 Wanda bug.
    """

class GradientCollector:
    """
    For Taylor scoring: collects gradients ∂L/∂W for each weight.
    Requires one forward + backward pass on calibration data.
    
    Usage:
        collector = GradientCollector(model, target_layers)
        
        # Forward + backward on calibration batch (requires labels)
        for images, labels in calibration_loader:
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
        
        gradients = collector.get_gradients()  # {layer_name: grad_tensor}
    
    Methods:
        get_gradients() → dict[str, Tensor]
            Returns {layer_name: weight.grad} with shape matching weight.
    """
```

### 7.2 `pruning/scoring.py`

```python
"""
Four pruning scoring functions.
Each returns a score tensor with the SAME SHAPE as the weight matrix.
Higher score = MORE important = should be KEPT.
"""

def magnitude_score(weight: Tensor) -> Tensor:
    """
    Score = |W|
    
    Args: weight [out_features, in_features]
    Returns: score [out_features, in_features]
    """
    return weight.abs()

def wanda_score(weight: Tensor, activation_norm: Tensor) -> Tensor:
    """
    Score = |W| * ||X||_2
    
    Args:
        weight: [out_features, in_features]
        activation_norm: [in_features]  (from ActivationCollector)
    Returns:
        score: [out_features, in_features]
    
    Implementation:
        return weight.abs() * activation_norm.unsqueeze(0)
    """

def taylor_score(weight: Tensor, gradient: Tensor) -> Tensor:
    """
    Score = |W * ∂L/∂W|
    
    Args:
        weight: [out_features, in_features]
        gradient: [out_features, in_features] (from GradientCollector)
    Returns:
        score: [out_features, in_features]
    """
    return (weight * gradient).abs()

def random_score(weight: Tensor) -> Tensor:
    """
    Score = random uniform, same shape as weight.
    Use a fixed seed per experiment for reproducibility.
    """
    return torch.rand_like(weight)
```

### 7.3 `pruning/masking.py`

```python
"""
Apply pruning masks based on scores.
"""

def compute_mask(score: Tensor, sparsity: float) -> Tensor:
    """
    Create a binary mask (0 = pruned, 1 = kept) for a single layer.
    
    Args:
        score: [out_features, in_features] — importance scores
        sparsity: float in [0, 1] — fraction of weights to REMOVE
    
    Returns:
        mask: [out_features, in_features] — binary tensor (0s and 1s)
    
    Implementation:
        1. Flatten the score tensor.
        2. Find the threshold = the (sparsity * num_elements)-th smallest value.
        3. mask = (score >= threshold).float()
        4. Verify: mask.sum() / mask.numel() ≈ (1 - sparsity)
    """

def apply_masks(model, masks: dict[str, Tensor]):
    """
    Apply pre-computed masks to the model weights IN-PLACE.
    
    Args:
        model: nn.Module
        masks: {layer_name: mask_tensor}
    
    Implementation:
        For each (name, mask) in masks:
            layer = get_layer_by_name(model, name)
            layer.weight.data *= mask
    
    IMPORTANT: This modifies model weights permanently.
    To test multiple sparsity levels, always reload from checkpoint first.
    """

def compute_global_masks(model, scores: dict[str, Tensor], sparsity: float) -> dict[str, Tensor]:
    """
    Global pruning: pool all scores across all layers, find one global
    threshold, then create per-layer masks.
    
    This is more standard than per-layer pruning because it allows
    different layers to have different effective sparsities based on
    their score distributions.
    
    Args:
        model: nn.Module (not modified)
        scores: {layer_name: score_tensor}
        sparsity: float — global fraction to remove
    
    Returns:
        masks: {layer_name: mask_tensor}
    """

def get_sparsity_stats(model, masks: dict[str, Tensor]) -> dict:
    """
    Report per-layer and global sparsity statistics.
    
    Returns:
        {
            'global_sparsity': float,
            'per_layer': {layer_name: {'sparsity': float, 'total': int, 'pruned': int}},
            'total_params': int,
            'nonzero_params': int
        }
    """
```

### 7.4 `pruning/layer_groups.py`

```python
"""
Group ViT layers by type for per-layer-type analysis (Experiment E5).
"""

def get_layer_groups(model) -> dict[str, list[str]]:
    """
    Returns a dict mapping layer-type names to lists of layer names.
    
    Returns:
        {
            'qkv': ['blocks.0.attn.qkv', 'blocks.1.attn.qkv', ...],
            'attn_out': ['blocks.0.attn.proj', 'blocks.1.attn.proj', ...],
            'mlp': ['blocks.0.mlp.fc1', 'blocks.0.mlp.fc2', ...],
            'patch_embed': ['patch_embed.proj']  (if included)
        }
    """

def prune_only_group(model, scores, sparsity, group_name, layer_groups):
    """
    Prune ONLY layers belonging to a specific group at the given sparsity.
    All other layers remain dense.
    
    Used for per-layer-type breakdown (E5).
    """
```

### 7.5 `pruning/nonuniform.py`

```python
"""
Pillar 2: HALM-inspired sensitivity-guided non-uniform sparsity allocation.
"""

def compute_layer_sensitivity(model, val_loader, criterion, device, scores_dict, layer_names):
    """
    For each layer, prune it alone at 50% sparsity, measure accuracy drop.
    
    Args:
        model: fine-tuned model (will be copied, not modified)
        val_loader: validation DataLoader
        criterion: loss function
        device: torch.device
        scores_dict: pre-computed scores (e.g., magnitude) for each layer
        layer_names: list of layer names to test
    
    Returns:
        sensitivity: dict[str, float] — {layer_name: balanced_acc_drop}
        Lower values (bigger drop) = more sensitive = needs less pruning.
    
    Implementation:
        For each layer:
            1. Deep copy the model
            2. Prune only that layer at 50% using magnitude scoring
            3. Evaluate balanced accuracy on validation set
            4. sensitivity[layer] = baseline_balanced_acc - pruned_balanced_acc
    """

def allocate_sparsity(sensitivity: dict[str, float], target_avg: float = 0.5, 
                       bins: dict = None) -> dict[str, float]:
    """
    Assign per-layer sparsity based on sensitivity scores.
    
    Default bins (from config):
        low_sensitivity (small acc drop):    sparsity = 0.7
        medium_sensitivity:                  sparsity = 0.5
        high_sensitivity (big acc drop):     sparsity = 0.3
    
    Bin boundaries: split layers into 3 equal groups by sensitivity rank.
    
    Adjust all sparsities proportionally so the weighted average 
    (by parameter count) equals target_avg.
    
    Args:
        sensitivity: {layer_name: acc_drop}
        target_avg: target average sparsity (0.5)
        bins: {low: 0.7, medium: 0.5, high: 0.3}
    
    Returns:
        allocation: {layer_name: float_sparsity}
    """

def apply_nonuniform_pruning(model, scores_dict, allocation):
    """
    Apply non-uniform sparsity using per-layer allocation.
    Each layer is pruned at its individually assigned sparsity level.
    
    Args:
        model: nn.Module (will be modified in-place)
        scores_dict: {layer_name: score_tensor}
        allocation: {layer_name: sparsity_float}
    
    Returns:
        masks: {layer_name: mask_tensor}
    """
```

---

## 8. QUANTIZATION MODULE

### 8.1 `quantization/ptq.py`

```python
"""
Pillar 3: INT8 Post-Training Quantization.
Applied on top of pruned models to test compression stacking.
"""

def quantize_model_dynamic(model):
    """
    Apply dynamic INT8 quantization to all Linear layers.
    
    Implementation:
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    Returns:
        quantized_model: the quantized model (CPU only for dynamic quant)
    """

def quantize_model_static(model, calibration_loader, device):
    """
    Apply static INT8 quantization with calibration.
    More accurate than dynamic, but requires calibration data.
    
    Steps:
        1. model.eval()
        2. Set qconfig: model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        3. Prepare: torch.ao.quantization.prepare(model, inplace=True)
        4. Calibrate: forward pass on calibration_loader (no grad)
        5. Convert: torch.ao.quantization.convert(model, inplace=True)
    
    NOTE: Static quantization may not work cleanly with ViT architectures
    due to LayerNorm and attention operations. If it fails, fall back to
    dynamic quantization and note this in the report.
    
    Returns:
        quantized_model
    """

def get_quantized_model_size(model) -> float:
    """
    Returns model size in KB after quantization.
    
    Implementation:
        Save model to a temporary file, read file size.
        Or: sum(p.nelement() * p.element_size() for p in model.parameters())
        For quantized: parameters are int8 (1 byte each) plus scale/zero_point overhead.
    """
```

---

## 9. KNOWLEDGE DISTILLATION MODULE

### 9.1 `models/distillation.py`

```python
"""
Pillar 4: Knowledge Distillation from DeiT-Small (teacher) to DeiT-Tiny (student).
"""

class DistillationLoss(nn.Module):
    """
    Combined loss: alpha * KL_div(soft_student, soft_teacher) + (1 - alpha) * CE(student, labels)
    
    Args:
        temperature: float (default 4.0) — softmax temperature
        alpha: float (default 0.7) — weight for soft loss
    
    Forward:
        student_logits: [batch, num_classes]
        teacher_logits: [batch, num_classes]
        labels: [batch]
        
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return alpha * soft_loss + (1 - alpha) * hard_loss
    """

def train_distillation(teacher, student, train_loader, val_loader, config, device):
    """
    Train student via KD from teacher.
    
    Args:
        teacher: fine-tuned DeiT-Small (frozen, eval mode)
        student: DeiT-Tiny (to be trained)
        train_loader, val_loader: DataLoaders
        config: distillation config dict
        device: torch.device
    
    Steps:
        1. teacher.eval(), freeze all teacher params
        2. student.train()
        3. For each epoch:
            a. For each batch:
                - Forward both teacher and student
                - Compute DistillationLoss
                - Backward, optimizer step
            b. Evaluate on val_loader: per-class sensitivity, balanced acc
            c. Save best checkpoint by balanced accuracy
        4. Return best student checkpoint
    
    Returns:
        student: trained DeiT-Tiny
        history: dict with per-epoch metrics
    """
```

---

## 10. EVALUATION MODULE

### 10.1 `evaluation/metrics.py`

```python
"""
Clinical evaluation metrics. This is the core differentiator of the project.
"""

def evaluate_model(model, data_loader, device, class_names=None):
    """
    Full evaluation with clinical metrics.
    
    Args:
        model: nn.Module (eval mode)
        data_loader: validation DataLoader
        device: torch.device
        class_names: list of 7 class names for readability
    
    Returns:
        results: dict with:
            'overall_accuracy': float,
            'balanced_accuracy': float,
            'per_class_sensitivity': dict[str, float],   # recall per class
            'per_class_specificity': dict[str, float],
            'per_class_f1': dict[str, float],
            'per_class_precision': dict[str, float],
            'confusion_matrix': np.ndarray [7, 7],
            'melanoma_sensitivity': float,   # shortcut to mel recall
            'bcc_sensitivity': float,        # shortcut to bcc recall
            'predictions': np.ndarray,       # for further analysis
            'ground_truth': np.ndarray
    
    Implementation:
        1. Forward all batches, collect predictions and labels
        2. Use sklearn.metrics:
           - accuracy_score
           - balanced_accuracy_score
           - classification_report (per-class P/R/F1)
           - confusion_matrix
        3. Per-class sensitivity = TP / (TP + FN) per class
    
    IMPORTANT:
        - melanoma_sensitivity is the MOST IMPORTANT metric.
        - Report it prominently in all results.
    """

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DANGEROUS_CLASSES = ['mel', 'bcc', 'akiec']  # clinically critical
```

### 10.2 `evaluation/latency.py`

```python
"""
Inference latency measurement.
"""

def measure_latency(model, input_shape=(1, 3, 224, 224), device='cpu',
                    warmup=10, timed_runs=100):
    """
    Measure inference latency with proper timing.
    
    For GPU:
        - Use torch.cuda.synchronize() before and after timing
        - Use torch.cuda.Event for precise timing
    For CPU:
        - Use time.perf_counter()
    
    Args:
        model: nn.Module (eval mode)
        input_shape: tuple
        device: 'cpu' or 'cuda'
        warmup: int — untimed warmup runs
        timed_runs: int — timed runs
    
    Returns:
        {
            'median_ms': float,
            'mean_ms': float,
            'std_ms': float,
            'p95_ms': float,
            'all_times_ms': list[float]
        }
    """
```

### 10.3 `evaluation/model_size.py`

```python
"""
Model size measurement.
"""

def get_model_size_kb(model, sparse=False):
    """
    Calculate model size in KB.
    
    If sparse=False: sum of all parameter sizes (float32 = 4 bytes each)
    If sparse=True: count only non-zero parameters
    
    Also supports saving to tmp file and reading file size for quantized models.
    
    Returns:
        {
            'total_params': int,
            'nonzero_params': int,
            'size_kb': float,           # actual storage
            'dense_size_kb': float,     # if all params were dense float32
            'compression_ratio': float  # dense / actual
        }
    """
```

---

## 11. EXPERIMENT SCRIPTS

### 11.1 `experiments/e1_finetune.py`

```
EXPERIMENT E1: Fine-tune DeiT-Tiny and DeiT-Small on HAM10000.

INPUT: 
  - Pretrained DeiT-Tiny and DeiT-Small from timm
  - HAM10000 dataset with stratified 80/20 split

PROCESS:
  1. Load model with load_deit_model(model_name, num_classes=7)
  2. Compute class weights via compute_class_weights()
  3. Loss = nn.CrossEntropyLoss(weight=class_weights)
  4. Optimizer = AdamW(lr=1e-4, weight_decay=0.05)
  5. Scheduler = CosineAnnealingLR with warmup
  6. Train for 20 epochs
  7. Save best checkpoint by balanced accuracy
  8. Save training history (loss, acc, balanced_acc per epoch)

OUTPUT:
  - results/checkpoints/deit_tiny_ham10000.pth
  - results/checkpoints/deit_small_ham10000.pth
  - results/logs/finetune_history.csv

RUNTIME: ~15-30 min per model on A100.
```

### 11.2 `experiments/e4_pruning_matrix.py`

```
EXPERIMENT E4: Main pruning matrix — 4 criteria × 5 sparsities × 2 models.

INPUT:
  - Fine-tuned DeiT-Tiny and DeiT-Small checkpoints
  - Calibration set (128 images from training split)
  - Validation set

PROCESS:
  For each model in [deit_tiny, deit_small]:
    1. Load fine-tuned checkpoint
    2. Run calibration pass → collect activation norms (for Wanda)
    3. Run calibration pass with gradient → collect gradients (for Taylor)
    4. For each criterion in [magnitude, wanda, taylor, random]:
      5. For each sparsity in [0.2, 0.4, 0.5, 0.6, 0.7]:
        a. RELOAD model from checkpoint (fresh copy each time!)
        b. Compute scores using the chosen criterion
        c. Compute global masks at target sparsity
        d. Apply masks to model
        e. Evaluate: overall_acc, balanced_acc, per_class_sensitivity
        f. Log all results to CSV

CRITICAL: Always reload the model from checkpoint before each (criterion, sparsity) 
combination. Never prune an already-pruned model.

OUTPUT:
  - results/logs/pruning_matrix.csv with columns:
    model, criterion, sparsity, overall_acc, balanced_acc, 
    mel_sensitivity, bcc_sensitivity, akiec_sensitivity, 
    nv_sensitivity, bkl_sensitivity, df_sensitivity, vasc_sensitivity,
    nonzero_params, size_kb

TOTAL RUNS: 4 × 5 × 2 = 40 evaluations.
RUNTIME: ~5-10 min each (inference only) = ~3-7 hours total.
```

### 11.3 `experiments/e5_perlayer_breakdown.py`

```
EXPERIMENT E5: Per-layer-type pruning breakdown at 50% sparsity.

INPUT:
  - Fine-tuned DeiT-Small checkpoint
  - Calibration data

PROCESS:
  For each layer_type in [qkv, attn_out, mlp, patch_embed]:
    For each criterion in [wanda, magnitude]:
      1. Reload model from checkpoint
      2. Prune ONLY layers of this type at 50% sparsity (others stay dense)
      3. Evaluate: balanced_acc, per_class_sensitivity
      4. Log results

OUTPUT:
  - results/logs/perlayer_breakdown.csv with columns:
    layer_type, criterion, balanced_acc, mel_sensitivity, bcc_sensitivity, ...

TOTAL RUNS: 4 layer_types × 2 criteria = 8 evaluations.
```

### 11.4 `experiments/e7_e10_nonuniform.py`

```
EXPERIMENT E7-E10: Sensitivity-guided non-uniform allocation (Pillar 2).

INPUT:
  - Fine-tuned DeiT-Small checkpoint
  - Per-layer sensitivity data (from E5, or computed fresh here)

PROCESS:
  1. Compute per-layer sensitivity scores using compute_layer_sensitivity()
  2. Allocate sparsity using allocate_sparsity(target_avg=0.5)
  3. For each criterion in [wanda, magnitude]:
     a. Apply non-uniform pruning using the allocation
     b. Evaluate: balanced_acc, per_class_sensitivity
  4. Compare against uniform 50% results from E4.

OUTPUT:
  - results/logs/nonuniform_allocation.csv
  - results/logs/layer_sensitivity_scores.csv (sensitivity per layer)

TOTAL RUNS: 2 criteria × 2 conditions (uniform/nonuniform) = 4 evaluations.
(Uniform results reused from E4.)
```

### 11.5 `experiments/e11_e13_quantization.py`

```
EXPERIMENT E11-E13: Post-pruning INT8 quantization (Pillar 3).

INPUT:
  - Best Wanda-pruned DeiT-Small at 50% sparsity (from E4)
  - Best magnitude-pruned DeiT-Small at 50% sparsity (from E4)

PROCESS:
  For each pruned_model in [wanda_50, magnitude_50]:
    1. Load pruned model
    2. Apply INT8 dynamic quantization
    3. Evaluate on CPU: balanced_acc, per_class_sensitivity
    4. Measure model size in KB
    5. Measure CPU inference latency

  Also evaluate:
    - Dense (unpruned, unquantized) baseline
    - Quantized-only (no pruning) baseline

OUTPUT:
  - results/logs/quantization_stacking.csv with columns:
    model_config, balanced_acc, mel_sensitivity, size_kb, latency_ms

TOTAL RUNS: 4 evaluations (dense, quant-only, prune+quant wanda, prune+quant mag).
```

### 11.6 `experiments/e14_e16_distillation.py`

```
EXPERIMENT E14-E16: KD pre-treatment before pruning (Pillar 4).

INPUT:
  - Fine-tuned DeiT-Small as teacher
  - DeiT-Tiny pretrained on ImageNet (not yet fine-tuned on HAM10000)

PROCESS:
  1. Run KD: teacher=DeiT-Small → student=DeiT-Tiny
     Config: T=4, alpha=0.7, 20 epochs, AdamW, cosine LR
  2. Now we have 3 DeiT-Tiny variants:
     A. Fine-tuned directly (from E1)
     B. Distilled from DeiT-Small (from step 1)
     C. ImageNet-only pretrained (no HAM10000 training — lower bound control)
  3. Prune each variant with Wanda at 50% sparsity
  4. Evaluate all 3 unpruned + all 3 pruned = 6 evaluations
  5. Compare per-class sensitivity

OUTPUT:
  - results/checkpoints/deit_tiny_distilled.pth
  - results/logs/kd_pretreatment.csv with columns:
    variant (direct/distilled/imagenet_only), pruned (yes/no), 
    balanced_acc, mel_sensitivity, bcc_sensitivity, ...

TOTAL RUNS: 6 evaluations + 1 training run (~30 min).
```

---

## 12. PLOTTING SPECIFICATIONS

### 12.1 `plotting/style.py`

```python
"""
Shared plot styling for consistent figures.
All plots should use this configuration.
"""

STYLE = {
    'figure_size': (8, 5),           # inches
    'dpi': 300,
    'font_family': 'serif',
    'font_size': 12,
    'title_size': 14,
    'legend_size': 10,
    'line_width': 2.0,
    'marker_size': 8,
    'grid_alpha': 0.3,
    'save_format': 'pdf',            # vector format for report
    'save_format_pres': 'png',       # raster for slides
}

# Color scheme — colorblind-friendly
CRITERION_COLORS = {
    'magnitude': '#1f77b4',   # blue
    'wanda':     '#d62728',   # red
    'taylor':    '#2ca02c',   # green
    'random':    '#7f7f7f',   # gray
}

CRITERION_MARKERS = {
    'magnitude': 'o',
    'wanda':     's',
    'taylor':    '^',
    'random':    'x',
}

CLASS_COLORS = {
    'mel':   '#d62728',    # red (dangerous)
    'bcc':   '#ff7f0e',    # orange (dangerous)
    'akiec': '#e377c2',    # pink (dangerous)
    'bkl':   '#2ca02c',    # green (benign)
    'nv':    '#1f77b4',    # blue (benign, majority)
    'df':    '#9467bd',    # purple
    'vasc':  '#8c564b',    # brown
}
```

### 12.2 Figure specifications

```
FIGURE 1 — Melanoma sensitivity vs sparsity (HEADLINE)
  Type: Line plot
  X-axis: Sparsity [0.2, 0.4, 0.5, 0.6, 0.7]
  Y-axis: Melanoma sensitivity (recall) [0, 1]
  Lines: 4 lines, one per pruning criterion (use CRITERION_COLORS)
  Subplots: 1×2, one per model (DeiT-Tiny, DeiT-Small)
  Add horizontal dashed line at baseline (0% sparsity) melanoma sensitivity.
  Add shaded region below 0.5 sensitivity labeled "clinically unacceptable".
  Title: "Melanoma Detection Rate Under Pruning"
  Save: results/figures/fig1_melanoma_sensitivity.pdf

FIGURE 2 — Balanced accuracy vs sparsity
  Type: Line plot (same layout as Figure 1 but with balanced accuracy)
  Title: "Balanced Accuracy Under Pruning"
  Save: results/figures/fig2_balanced_accuracy.pdf

FIGURE 3 — Per-layer-type sensitivity breakdown
  Type: Grouped bar chart
  X-axis: Layer types [QKV, Attn Output, MLP, Patch Embed]
  Y-axis: Balanced accuracy drop from dense baseline
  Groups: Wanda vs Magnitude (2 bars per layer type)
  Title: "Which ViT Components Encode Diagnostic Features?"
  Save: results/figures/fig3_perlayer_bars.pdf

FIGURE 4 — Non-uniform vs uniform allocation
  Type: Grouped bar chart
  X-axis: 7 class names
  Y-axis: Per-class sensitivity
  Groups: 3 bars per class (dense baseline, uniform 50%, non-uniform avg-50%)
  Title: "Sensitivity-Guided Allocation vs Uniform Pruning"
  Save: results/figures/fig4_nonuniform.pdf

FIGURE 5 — Compression stacking
  Type: Grouped bar chart
  X-axis: 7 class names
  Y-axis: Per-class sensitivity
  Groups: 3 bars per class (dense, pruned-only, pruned+quantized)
  Title: "Does Stacking Quantization on Pruning Break Diagnostic Safety?"
  Save: results/figures/fig5_stacking.pdf

FIGURE 6 — KD pre-treatment
  Type: Grouped bar chart
  X-axis: 7 class names
  Y-axis: Per-class sensitivity (after Wanda 50% pruning)
  Groups: 3 bars (direct fine-tune, distilled, ImageNet-only)
  Title: "Does Knowledge Distillation Produce Pruning-Robust Features?"
  Save: results/figures/fig6_kd_pretreatment.pdf
```

---

## 13. MASTER EXPERIMENT RUNNER

### `experiments/run_all.py`

```
Usage:
  python experiments/run_all.py --config configs/default.yaml --pillars 1 2 3 4

Executes experiments in order:
  Pillar 0 (setup):    E1 (finetune), E2 (baseline eval), E3 (calibration)
  Pillar 1 (pruning):  E4 (pruning matrix), E5 (per-layer), E6 (diagnostic safety)
  Pillar 2 (HALM):     E7-E10 (non-uniform allocation)
  Pillar 3 (quant):    E11-E13 (INT8 stacking)
  Pillar 4 (KD):       E14-E16 (distillation pre-treatment)

Each pillar can be run independently. Results accumulate in results/logs/.
All experiments are idempotent (can be re-run safely).
```

---

## 14. KEY IMPLEMENTATION NOTES FOR CODEX

1. **ALWAYS reload from checkpoint before each pruning experiment.** Never prune an already-pruned model. The pattern is: load checkpoint → compute scores → compute masks → apply masks → evaluate → discard model.

2. **Wanda hooks must capture INPUT activations, not outputs.** In `register_forward_hook(fn)`, the function signature is `fn(module, input, output)`. Use `input[0]` (the input tensor), NOT `output`.

3. **Sanity check at 0% sparsity.** Before running any pruning experiments, verify that the evaluation pipeline at 0% sparsity produces EXACTLY the same metrics as the baseline evaluation (E2). If off by more than 0.1%, there is a bug in the masking or evaluation code.

4. **Class imbalance handling.** HAM10000 is heavily imbalanced (~67% melanocytic nevus). Use weighted cross-entropy for training. Use BALANCED accuracy (not overall accuracy) as the primary aggregate metric. Report per-class sensitivity prominently.

5. **Exclude classifier head and patch embedding from pruning by default.** These are small layers whose pruning disproportionately damages accuracy. Note this decision explicitly in code comments.

6. **Random pruning needs a fixed seed per experiment for reproducibility.** Set `torch.manual_seed(42)` before generating random scores, and reset it for each experiment.

7. **INT8 quantization runs on CPU only** (for dynamic quantization via torch.ao). Evaluate quantized models on CPU and measure CPU latency.

8. **All CSV logs should include a timestamp column** so experiments can be tracked chronologically.

9. **Use `torch.no_grad()` for all inference-only experiments** (pruning evaluation, quantization evaluation). Only E1 (finetune), E14 (KD training), and Taylor scoring (needs one backward pass) require gradients.

10. **Save masks alongside results.** For each (model, criterion, sparsity) combination, save the mask dict to disk so experiments can be reproduced without recomputing scores.
