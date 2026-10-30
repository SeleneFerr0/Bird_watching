# Birder 0.9.0: algorithm research and implementation decisions

Research date: October 29, 2026. Intended use: bird photographs and mixed wildlife images.

See the [validation record](validation-0.9.0.md) for executed checks and their limits.

## Recommendation

Use a detector to propose animals, classify each padded crop with a species model,
and retain ambiguous identifications for review. Compare a lightweight YOLO model
with Birder's existing transformer detectors on the same photographs. Enable
overlapping tiles when subjects are small relative to the source image.

This is a testable engineering hypothesis, not a measured improvement in wildlife
accuracy. The folder contains no labeled field dataset or local trained weights.
No training, fine-tuning, or ecological accuracy benchmark is claimed for 0.9.0.

## What the existing project already contains

The 0.8.4 source includes D-FINE, RF-DETR, RT-DETR v1/v2, Deformable DETR,
Faster R-CNN, FCOS, and YOLO v2/v3/v4. It already has native tiled inference,
weighted box fusion, detection augmentation, self-supervised training, and BioCLIP
2 / 2.5 image encoders. Simply adding a newer architecture would leave the main
workflow gap: there was no unified path from animal boxes to per-animal species
predictions and a human-review decision.

Existing BioCLIP conversions are **image feature encoders**, not complete
zero-shot species classifiers. A species head or the original text encoder and
candidate taxonomy are required. Applying softmax to embedding dimensions does
not turn them into species probabilities.

## Detection options

| Option | Evidence and practical tradeoff | Decision for this release |
| --- | --- | --- |
| Original YOLOv5 | Mature baseline; original repository checkpoints use a different loading API from the current Ultralytics package. | Do not present it as the newest architecture or silently load incompatible checkpoints. |
| YOLOv5u | Ultralytics' updated v5 family uses an anchor-free head and the current `YOLO` API. | Optional backend, with `yolov5nu.pt` / `yolov5su.pt` as reproducible starting points. |
| YOLO26 | Current Ultralytics family with an optional end-to-end detection head. Vendor COCO speed/accuracy numbers describe particular hardware and export settings, not this wildlife application. | Supported through the same optional adapter, e.g. `yolo26n.pt`. Benchmark locally before selecting a default. |
| D-FINE / RF-DETR | Modern detector families already implemented in Birder. Their upstream projects provide reference implementations and detection results. | Keep native support and expose it in the same crop-classification pipeline. No new training is needed to connect existing weights. |
| MegaDetector / SpeciesNet | Camera-trap systems illustrate the value of separating general animal localization from species identification. Their domain differs from nature photography. | Architectural references and future comparison candidates, not additional dependencies in 0.9.0. |

Sources: [YOLOv5u compatibility and architecture](https://docs.ultralytics.com/models/yolov5),
[YOLO26](https://docs.ultralytics.com/models/yolo26),
[D-FINE](https://github.com/Peterande/D-FINE),
[RF-DETR](https://github.com/roboflow/rf-detr),
[MegaDetector](https://github.com/agentmorris/MegaDetector),
[SpeciesNet](https://github.com/google/cameratrapai).

Ultralytics currently distributes its software/models under AGPL-3.0 and
Enterprise terms. The adapter is optional and does not copy its implementation
or bundle its weights. Installing it does not remove those upstream obligations.
Check model-weight and dataset terms separately for all detector families.
This release makes no licensing change to existing Birder source.

## Small subjects: resolution before model size

A distant bird can lose identifying details when a large photograph is resized
to the detector's input size. Slicing runs the detector on overlapping crops,
maps boxes back into original coordinates, and merges duplicate predictions.
SAHI provides published evidence for this mechanism on aerial detection
benchmarks, which is motivation to test it on bird images rather than proof of a
bird-specific gain. [SAHI paper](https://arxiv.org/abs/2202.06934)

0.9.0 reuses Birder's existing window generation and coordinate mapping for both
native and YOLO backends. It includes a full-image pass to preserve large
subjects, then uses class-aware NMS to suppress overlap duplicates. Tiles run
sequentially and return CPU tensors. This trades additional detector work for
resolution without retaining the whole survey on the accelerator.

Potential regressions must be measured: duplicates that survive merging,
neighboring birds suppressed by NMS, clipped animals at tile edges, and increased
latency. Crop padding and overlap are configurable; they are not universal
constants. The default crop padding adds 10% of box width/height to each side.

## Classification options

| Option | Fit for the task | Decision |
| --- | --- | --- |
| Existing regional species classifier | Named output classes, lower deployment complexity, known geographical scope. | Immediately usable with the new pipeline. Reject classifiers without a named head. |
| BioCLIP 2 | Biology-focused pretraining and taxonomic supervision make it a relevant transfer-learning candidate. | Use existing Birder encoders for linear probing and fine-tuning on actual species labels. |
| BioCLIP 2.5 Huge | Upstream reports improved biological recognition with a larger ViT-H/14 encoder and greater inference requirements. | Include in the benchmark shortlist; do not make a large feature extractor the CPU default or claim its embeddings are species scores. |
| DINOv3 | General visual features provide another transfer-learning candidate, with separate model-access/license terms. | Future controlled comparison; not a new pretrained species model in this release. |

Sources: [BioCLIP 2 paper](https://arxiv.org/abs/2505.23883),
[BioCLIP 2.5 model card](https://huggingface.co/imageomics/bioclip-2.5-vith14/blob/main/README.md),
[DINOv3 model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md).

Crop classification can separate several animals in one photo and reduce
background dominance. It also introduces detector error and a distribution
shift from full-image classifier training. Include realistic detected crops,
imperfect boxes, and surrounding context during fine-tuning and calibration.
For well-framed single-bird photographs, full-image classification can remain
better. Always keep that baseline in the experiment.

## Confidence and review

0.9.0 uses temperature-scaled softmax species scores, a minimum top score, and a
minimum gap between the two leading candidates. Rejected identifications keep
their candidate names, scores, and review reasons; the accepted `species` field
is null. Detector confidence is kept separate from species confidence.

The new `fit_temperature` helper minimizes held-out negative log likelihood
using one positive temperature, including the unscaled baseline in selection.
Temperature scaling preserves rankings and addresses calibration rather than
top-1 accuracy. [Guo et al., ICML 2017](https://arxiv.org/abs/1706.04599)

Defaults of 0.7 probability and 0.15 margin are provisional operating settings.
They are not estimates of field reliability. Fit temperature and select review
thresholds on a separate calibration set, then freeze them before evaluating the
test set. Unknown species can still receive high confidence: thresholding alone
is not an out-of-distribution detector. Do not label `no_detections` as verified
absence, or interpret `ok` as expert confirmation.

## Reproducible field evaluation

1. Create independently labeled photographs with boxes and species, including
   empty scenes, multiple subjects, unfamiliar species, and difficult examples.
   Define the geographical and taxonomic scope before training.
2. Deduplicate original photographs and bursts. Split by capture event,
   photographer/site, or camera, according to the deployment question. Grouped
   splitting keeps related observations together; it does not automatically
   identify those relationships. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
3. Keep training, model-selection, calibration, and final-test partitions
   separate. Include site/time holdouts when evaluating transfer to new surveys.
4. Run the same classifier on full images, detector crops, and tiled-detector
   crops. Compare YOLOv5u, YOLO26, and a native detector using identical input
   sets, species mappings, score thresholds, and hardware.
5. Report detection AP/recall, small-subject recall, per-species precision/recall
   and macro-F1. Also report end-to-end species recall over **all labeled
   subjects**, so missed detections do not disappear from the denominator.
6. Measure accepted-prediction error against coverage/review rate. Report false
   species sightings, missed rare species, calibration log loss, latency
   percentiles, memory, model size, hardware, batch size, and input resolution.
7. Inspect errors before selecting a winner. Save model identifiers, package
   versions, exact weight hashes, split manifests, and all inference settings.

Suggested experiment matrix:

| Run | Localization | Classification | Main question |
| --- | --- | --- | --- |
| A | Whole image | Current named classifier | Existing behavior baseline |
| B | Native Birder detector | Same classifier on padded crops | Effect of localization |
| C | YOLOv5u | Same crop classifier | Familiar YOLO baseline |
| D | YOLO26 | Same crop classifier | Newer detector comparison |
| E | Best detector with tiles | Same crop classifier | Small-subject benefit and latency cost |
| F | Selected detector | Fine-tuned BioCLIP species head | Value of biological pretraining |

There are no numeric field results for these runs yet. Repository unit tests
verify software behavior, not ecological identification accuracy.

## Shipped versus deferred

Implemented: Unicode path handling without renaming, consistent single-image
color modes and file lifetimes, native/optional YOLO adapters, padded crop
classification, shared tiled detection, selective predictions, temperature
fitting, bounded crop batching, and streaming UTF-8 JSONL with run settings.

Deferred pending labeled data: training new detector or classifier weights,
choosing a model winner, calibrated operating thresholds, zero-shot taxonomy/text
inference, learned unknown-species rejection, and video tracking. No upstream
public package, model registry, or release tag is published by these local edits.
