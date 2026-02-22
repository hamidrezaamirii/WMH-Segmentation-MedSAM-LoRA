## Project Overview

White matter hyperintensities (WMH) are radiological markers of cerebral
small vessel disease, strongly associated with cognitive decline, stroke
risk, and progression to vascular dementia. Accurate quantification of WMH
volume and spatial distribution from routine MRI is essential for clinical
trials, longitudinal patient monitoring, and population-level
epidemiological studies. However, manual segmentation remains the clinical
standard at most centres — a process that is time-intensive,
poorly reproducible across raters, and impractical at scale.

This project presents **WMH-LoRA**, a parameter-efficient approach to
automated WMH segmentation that adapts Med-SAM, a vision foundation model
pre-trained on over 1.5 million medical images, to the domain of cerebral
white matter lesion delineation. Rather than training a task-specific
network from scratch or fully fine-tuning a large model, we apply
**Low-Rank Adaptation (LoRA)** to inject a small number of trainable
parameters into the frozen Med-SAM encoder. With rank-64 LoRA applied to
both attention and MLP layers, only **3.5% of the total model parameters
are updated during training**, while the remaining 96.5% retain the rich
visual representations learned during large-scale medical image
pre-training.

A critical design decision is the use of **FLAIR MRI as the sole input
modality**. While most competitive methods on the MICCAI 2017 WMH
Segmentation Challenge rely on co-registered FLAIR and T1-weighted pairs,
T1 scans are frequently unavailable in retrospective clinical datasets,
acquired with incompatible protocols, or degraded by motion artefacts.
A robust FLAIR-only pipeline substantially broadens clinical applicability.

Despite this single-modality constraint, WMH-LoRA achieves a **global
Dice coefficient of 0.78** across three heterogeneous acquisition sites
(Utrecht/Philips, Singapore/Siemens, GE3T/GE), placing it within 2–3
points of published multi-modal methods that use fully trained
architectures with an order of magnitude more trainable parameters. The
model further demonstrates strong **volumetric agreement** with expert
annotations (R² = 0.963, mean bias = −0.693 mL), confirming its
suitability for quantitative neuroimaging workflows where lesion volume
serves as a primary endpoint.

The training pipeline incorporates several techniques designed to address
the specific challenges of WMH segmentation: a **multi-scale decoder with
deep supervision** to recover fine boundary detail of small periventricular
and deep white matter lesions; a **composite loss function** combining
Focal Loss, Soft Dice Loss, and a morphology-based Boundary Loss to handle
extreme foreground-background class imbalance; **CosineAnnealingWarmRestarts**
scheduling over 100 epochs to escape local minima; and **8-fold test-time
augmentation** with optimised decision thresholding for inference.

This work demonstrates that foundation model adaptation via LoRA is a
viable and practical paradigm for clinical neuroimaging tasks — achieving
competitive segmentation accuracy at a fraction of the computational and
data cost traditionally required.


## Clinical Significance

The clinical value of automated WMH segmentation extends across three
domains where this work makes a direct contribution.

**Longitudinal disease monitoring.** In clinical practice, WMH burden is
tracked over time to assess disease progression in patients with
hypertension, diabetes, or early-stage cognitive impairment. Accurate
automated volumetry enables detection of subtle changes between timepoints
that fall below the threshold of visual assessment. The volumetric
agreement demonstrated by WMH-LoRA (R² = 0.963, mean bias < 1 mL) falls
within the range of inter-rater variability reported for expert manual
segmentation, suggesting the model could serve as a consistent and
reproducible surrogate for manual annotation in longitudinal studies.

**Clinical trial endpoint quantification.** Pharmaceutical trials
targeting cerebral small vessel disease increasingly use WMH volume change
as a primary or secondary endpoint. Automated segmentation eliminates
rater-dependent variability and enables centralised, blinded analysis of
multi-site trial data. The cross-site generalisation demonstrated in this
work — maintaining Dice > 0.75 across Philips, Siemens, and GE scanners
without site-specific fine-tuning — addresses one of the principal
barriers to deploying automated methods in multi-centre studies.

**Accessibility and deployment.** The FLAIR-only design is deliberately
chosen for maximum clinical applicability. FLAIR is the single most
commonly acquired sequence in neurological MRI protocols worldwide. By
removing the T1 co-registration requirement, WMH-LoRA can be applied to
legacy datasets, emergency department scans acquired with abbreviated
protocols, and resource-limited settings where multi-sequence acquisitions
are not routine. The parameter-efficient architecture further reduces the
computational requirements for deployment — inference runs on a single
consumer GPU, and the trainable checkpoint is under 25 MB, enabling
practical integration into hospital PACS/AI pipelines.

**Limitations and future directions.** The absence of T1 input means the
model cannot leverage grey-white matter contrast to disambiguate WMH from
cortical lesions or enlarged perivascular spaces, which likely accounts for
the performance gap relative to multi-modal methods on cases with atypical
lesion patterns. Performance on subjects with very low lesion burden
(< 2 mL total WMH) remains an open challenge, as small, sparse lesions
occupy fewer than 0.01% of brain voxels. Future work will explore T1
integration as an optional auxiliary channel, 3D contextual encoding to
capture inter-slice spatial coherence, and evaluation on external cohorts
(e.g., UK Biobank, ADNI) to assess out-of-distribution robustness.
