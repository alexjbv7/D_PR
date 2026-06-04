# ADR-034: ResMLPClassifier reemplaza DeepMLPClassifier en multi-horizon trainer

**Status:** Proposed
**Date:** 2026-05-26
**Deciders:** Alex (lead), Claude Cowork (assistant)
**Affects:** `research/models/zoo.py`, `research/models/multi_horizon/trainer.py`, calibration cascade
**Supersedes:** parte de la jerarquía ML descrita en `CLAUDE.md` §6.3

## Contexto

El `DeepMLPClassifier` actual (MLP plano con `[Linear → ReLU → Dropout] × N`)
es uno de los 3 modelos del `MultiHorizonTrainer` introducido en ADR-028.
Aunque cumple el contrato `BaseModel.predict_proba()`, presenta tres
limitaciones cuantificables en horizontes intraday/swing/daily con bajo SNR:

1. **Degradación de gradientes** en profundidad > 3 capas sin skip connections.
   Empíricamente, `bias_variance_gap` aumenta y `val_loss` se estanca.
2. **Activación ReLU** satura asimétricamente en features anti-simétricas
   (z-scores, log-returns). SwiGLU/GeGLU mejoran SNR en literature tabular
   (Shazeer 2020; Gorishniy et al. 2021).
3. **Miscalibración out-of-the-box.** `IsotonicCalibrator` solo absorbe
   parcialmente la overconfidence de NN; ECE sobre val OOS supera 0.07 en runs
   recientes, rompiendo el contrato del meta-labeler+bayesian sizer (< 0.05).

El multi-horizon trainer es el componente más sensible: su salida alimenta
directamente al allocator Thompson (ADR-032) y al confirmation gate (ADR-033).
Mejorar su calibración y bias-variance se traduce en mejor sizing y mejor
selection de horizontes activos.

## Decisión

Reemplazar `DeepMLPClassifier` por una nueva clase **`ResMLPClassifier`**
con la siguiente arquitectura, manteniendo `DeepMLPClassifier` disponible
como baseline A/B durante shadow trading:

```
Input(n_features)
  → NumericalEmbedding(d_model)          # Linear + LayerNorm
  → [ResBlock] × n_blocks
       ResBlock(x):
         y = LayerNorm(x)
         y = Linear(d, 2d)
         y = SwiGLU(y)                   # half-half split, gated
         y = Dropout(p)
         y = Linear(2d, d)
         return x + y
  → LayerNorm
  → Linear(d, n_classes)
  → Softmax
  → TemperatureScaling                   # fit en val fold, NUNCA test
  → IsotonicCalibrator                   # cascade (existente, no modificado)
```

Hiperparámetros con Optuna search anidado en walk-forward:
- `d_model ∈ {64, 128, 256}`, default 128
- `n_blocks ∈ {2, 4, 6}`, default 4
- `dropout ∈ {0.10, 0.15, 0.20}`, default 0.15
- `lr ∈ {1e-4, 3e-4, 1e-3}` con cosine schedule + 5% warmup
- `weight_decay ∈ {1e-5, 1e-4, 1e-3}`
- `batch_size ∈ {256, 512, 1024}`
- `label_smoothing = 0.05`
- early stopping `patience = 10`

## Alternativas consideradas

| Alternativa | Pros | Contras | Veredicto |
|---|---|---|---|
| **TabNet** (Arik & Pfister 2021) | Interpretable, sparse feature selection | Training más lento, hp sensible | Diferido a fase 6 como 4º modelo |
| **FT-Transformer** (Gorishniy 2021) | SOTA tabular 2022-2024 | Requiere ≥ 10⁵ muestras/horizonte | Diferido: dataset insuficiente en daily |
| **Mixture of Experts** | Capacidad alta sin coste lineal | Training inestable, hard to calibrate | Rechazado: riesgo operativo |
| **Mantener DeepMLP** (status quo) | Cero riesgo de regresión | Bias-variance gap y ECE confirmados como cuellos de botella | Rechazado |

ResMLP es la elección dominada en el plano (capacidad ganada vs riesgo añadido)
para el dataset y horizontes actuales.

## Criterios de aceptación (gating de promoción)

El nuevo modelo solo reemplaza DeepMLP en producción si, sobre walk-forward OOS
concatenado ≥ 30 folds:

- `ΔDSR_OOS ≥ +0.05` vs `DeepMLPClassifier` (statistically significant)
- `ECE ≤ 0.05` post-cascade (temperature → isotonic)
- `bias_variance_gap ≤ 0.10`
- `latency_p99_inference ≤ 100 ms` (CPU batch=1)
- `vram_train ≤ 8 GB` con batch=1024
- `no class collapse`: `min(per_class_predict) > 0.05`

Tras shadow trading 30 días en paralelo con DeepMLP durante el siguiente run
de paper:
- A/B comparison: B ≥ A en ≥ 4 de 5 métricas operativas (DSR, ECE, max DD,
  meta-label hit rate, latency p99).
- B no peor que A en ninguna métrica por > 5%.

## Pipeline de calibración cascada

1. `ResMLPClassifier` entrena con CE loss + label smoothing 0.05.
2. `TemperatureScaling` fitea `T` sobre val fold del walk-forward (NUNCA test).
3. `IsotonicCalibrator` existente se ajusta sobre `temperature_scaled_proba`,
   NO sobre raw logits.
4. ECE final se mide post-isotonic.

## Implementación y entrega

- **Ejecutor:** Claude Code Sonnet 4.6 (decisión revisable; Opus 4.6 si la
  search Optuna identifica patología no resuelta).
- **Coordinación:** Cowork (este modo) emite el prompt, audita con subagente.
- **Entregables:**
  - `research/models/nn_layers.py` (nuevo)
  - `research/models/zoo.py:ResMLPClassifier` (clase nueva)
  - `research/tests/test_resmlp_classifier.py` (≥ 12 tests)
  - `research/tests/test_no_leakage_multi_horizon.py` (+2 tests adversariales)
  - `benchmarks/resmlp_vs_deepmlp.md` (tabla A/B con números reales)
  - Este ADR (commiteado con la implementación)

## Contratos a respetar

- `BaseModel`: signatures `fit(X, y)`, `predict_proba(X)`, `save(path)`, `load(path)`
- `predict_proba` shape `(n, n_classes)`, sum=1 ± 1e-6
- Determinismo: `torch.use_deterministic_algorithms(True)`, cudnn benchmark off
- Anti-leakage: `T` (temperature) SOLO fit en val fold del WF
- NO modificar `IsotonicCalibrator`, `BaseModel`, ni `walk_forward_runner.py`

## Plan de rollout

1. Implementación en branch `feat/resmlp-replacement` (post paper run 2026-06-19).
2. Benchmarks históricos walk-forward → reporte interno.
3. Shadow trading 30 días paralelo con DeepMLP en siguiente paper run.
4. A/B verdict → promoción (canary 5% → 100% del slot de `deep_mlp`).
5. `DeepMLPClassifier` queda como `legacy/baseline` para auditoría.

## Consecuencias

**Positivas:**
- Mejor calibración esperada (ECE objetivo < 0.05 post-cascade).
- Bias-variance gap más bajo en horizontes con poco volumen (daily).
- Path de modernización abierto para TabNet/FT-Transformer en fase 6.

**Negativas:**
- Coste training mayor (GPU recomendada, fallback CPU con `n_blocks=2`).
- Optuna search más caro (~3x walltime).
- Más superficie de mantenimiento: nuevo módulo `nn_layers.py`.

**Neutrales:**
- DeepMLP queda en codebase como baseline (no eliminación inmediata).
- Pipeline calibración isotónica intacto — se añade temperature scaling en cascada.

## Referencias

- Shazeer (2020). *GLU Variants Improve Transformer*. arXiv:2002.05202.
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- He et al. (2015). *Deep Residual Learning for Image Recognition*. CVPR.
- Gorishniy et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS.
- Ioffe & Szegedy (2015). *Batch Normalization*. ICML.
- López de Prado (2018). *Advances in Financial Machine Learning*. Wiley.
