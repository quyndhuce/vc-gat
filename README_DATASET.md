# Decision-Certified VC-GAT: Cultural Heritage Video Captioning & Fuzzy Knowledge Graph Dataset

## 📌 Dataset Overview

This dataset accompanies the publication:  
**"Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning"**  
Submitted to *International Journal of Approximate Reasoning* (IJAR), Elsevier.

- **Anonymous Code Repository:** [https://anonymous.4open.science/r/vc-gat-E1C7/](https://anonymous.4open.science/r/vc-gat-E1C7/)
- **Dataset Archive (Figshare DOI):** [10.6084/m9.figshare.35113231](https://doi.org/10.6084/m9.figshare.35113231)
- **Dataset & Video Mirror (Google Drive):** [https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing](https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing)

The dataset contains the complete multimodal cultural heritage knowledge graph, verified scene-conditioned Horn evidence, contrastive fact pairs, and pre-extracted video scene graphs across Hanoi and Ho Chi Minh City cultural landmarks.

---

## 📂 Dataset Architecture & Directory Layout

```text
vc_gat_dataset/
├── README_DATASET.md                             # This documentation and reproduction specification
│
├── data/                                         # Core benchmark data and knowledge graphs
│   ├── caption_dataset/                          # Benchmark captions across epistemic regimes
│   │   ├── balanced_1000_captions_qwen25_14b.json # Hanoi-500 test captions (E1-E6 configurations)
│   │   ├── generated_captions_test_full_1119.jsonl
│   │   ├── generated_captions_val.jsonl
│   │   └── generated_captions_train.jsonl
│   ├── fkg_per_kb/                               # Fused Knowledge Graphs (4,093 cultural landmarks)
│   │   ├── fkg_global_merged.json                # Complete fused graph (13,553 facts, RDF/JSON-LD)
│   │   ├── fkg_global_merged.ttl                 # Turtle RDF serialization
│   │   ├── fkg_global_networkx.pkl               # NetworkX graph structure
│   │   └── <landmark_id>/                        # Individual landmark sub-graphs
│   ├── hcmc_benchmark/                           # Out-of-domain Zero-Shot transfer dataset (HCMC-300)
│   │   ├── hcmc_300_clips_manifest.json
│   │   ├── hcmc_eval_facts.jsonl
│   │   └── hcmc_predictions_table11.json
│   ├── unified_coco_dataset/                     # Grounding annotations and object bounding boxes
│   │   ├── annotations/
│   │   │   ├── train_coco.json                   # 3,227 video clips with bounding box annotations
│   │   │   ├── test_coco.json                    # 1,119 video clips with bounding box annotations
│   │   │   └── taxonomy_categories.json          # 285 visual universe concept categories
│   │   ├── test_video_durations.json             # Frame-accurate clip duration records
│   │   └── test_videos_duration_stats.json       # Distribution statistics (7.62 hours test duration)
│   ├── dataset_splits_scheme_A_parent.json       # Canonical Scheme A split (Parent-video disjoint)
│   ├── dataset_splits_video_level.json           # Secondary flat split
│   └── dataset_splits_stratified.json            # Stratified landmark split
│
├── scene_conditioned_gold/                       # Verified Horn reasoning and gold facts
│   ├── config/
│   │   ├── ontology_mapping.yaml                 # Formal Horn rules, predicates, attenuation lambda=0.90
│   │   └── models.yaml                           # System hyperparameters and threshold configurations
│   └── outputs/
│       ├── all_clips_manifest.json               # Master manifest for all 1,651 benchmark clips
│       ├── evaluation_clips_manifest.json        # Test-500 (Hanoi) + Test-300 (HCMC) manifest
│       ├── train_val_clips_manifest.json         # Train-702 + Val-149 manifest
│       ├── scene_conditioned_fgold_horn_complete.json # Full Horn complete gold standard facts
│       ├── scene_conditioned_gold_horn_complete.jsonl.gz # Gzipped line-delimited gold facts
│       ├── contrastive_pairs_horn_complete.jsonl # Hard-negative intra-landmark contrastive pairs
│       ├── contrastive_pairs_train_val_horn_complete.jsonl # Training contrastive challenge pairs
│       ├── normalized_facts_horn_complete.jsonl  # 16,347 normalized candidate facts
│       ├── landmark_candidates_pool_horn_complete.json # Per-landmark candidate fact pools
│       ├── fgold_facts_dictionary.json           # Ground-truth fact dictionary and predicates
│       └── fgold_test_evidence_audit_table.tsv   # Cryptographic audit table of all evidence
│
├── experiments/features/                         # Pre-extracted visual scene graphs for direct model input
│   ├── train_scene_graphs.pkl.gz                 # Compressed scene graphs for Train (702) + Val (149)
│   ├── test_scene_graphs.pkl                     # Pre-extracted scene graphs for Test Hanoi-500
│   ├── gcn_embeddings.npy                        # Initial entity & relation embeddings
│   ├── concept_embeddings.npy                    # Visual concept text embeddings (SigLIP 2)
│   └── scene_graph_dict.pkl                      # Mapping dictionary of visual objects and relations
│
└── legacy_archive/data_label_generation_pipeline/ # Silver reference and raw annotation audit logs
    ├── scene_conditioned_gold/reports/           # Dual-annotator agreement and verification logs
    └── two_level_silver_pipeline/                # Silver fact candidate records
```

---

## 📊 Benchmark Splits Specification

The benchmark enforces **Scheme A (Parent-Video Grouped Split)** where no two segments originating from the same parent source video appear in both training/calibration and evaluation splits:

| Split Identifier | Clip Count | Total Duration | Mean Clip Duration | Target Evaluation |
|:---|:---:|:---:|:---:|:---|
| `train` | 702 | 175.7 min (2.93 h) | 15.0 s | VC-GAT degree-debiased ranking & LoRA training |
| `val` | 149 | 35.4 min (0.59 h) | 14.3 s | Finite-sample Conformal Risk Control calibration |
| `test_balanced_500` | 500 | 203.4 min (3.39 h) | 24.4 s | Hanoi In-Domain Benchmark (Tables 2, 5, 6, 7) |
| `test_hcmc_300` | 300 | 80.2 min (1.34 h) | 16.0 s | HCMC Cross-Domain Zero-Shot Transfer (Table 11) |
| **Total Benchmark** | **1,651** | **494.7 min (8.25 h)** | **18.0 s** | **100% Parent-Video Disjointness** |

---

## 🔬 Reproduction with Official Codebase

To reproduce all tables and figures reported in the manuscript using this dataset:

1. Clone the official anonymous repository:
   ```bash
   git clone <ANONYMIZED_REPOSITORY_URL>
   cd vc-gat
   ```

2. Obtain the dataset package (`vc_gat_dataset_clean_462M.tar.gz` or Figshare package) from:
   - **Google Drive Mirror:** [https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing](https://drive.google.com/drive/folders/1iTEHWfA5XfYW5Y1KvoFMofbhSpki2Dwv?usp=sharing)
   - **Figshare DOI:** [10.6084/m9.figshare.35113231](https://doi.org/10.6084/m9.figshare.35113231)

   Extract the dataset archive into the repository root:
   ```bash
   tar -xzf vc_gat_dataset_clean_462M.tar.gz -C .
   ```

3. (Optional) To inspect or process the raw 4,346 cut video clips, download `videos/` (50.6 GB) from the Google Drive mirror into `data/unified_coco_dataset/videos/`.

3. Install CPU verification dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute the master reproduction suite:
   ```bash
   # Level 1 & Level 2 verification of all paper tables (Tables 2, 5, 6, 7, 11)
   python3 reproduce_all.py

   # Recompute 10,000 cluster bootstrap resamples
   python3 recompute_table22.py

   # Run automated test suite
   python3 -m unittest discover tests/
   ```

---

## 📄 License & Terms of Use

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share and adapt the material for any purpose, provided appropriate credit is given.
