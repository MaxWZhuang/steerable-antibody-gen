# steerable-antibody-gen
Antibody Language Model, implementing steerability with SAE-derived concept labelling

Overall plan for Phase 1: 

1. Pretrain with PyTorch on antibody sequence biology first on OAS (Observed Antibody Space)
  a. Due to limited size of dataset (1.3M), constrain to MLM rather than a full autoregressive system -- allows understanding of linking and being able to optimize on a smaller range (end goal here: adjust HDCR3, as that is primarily responsible for folding/binding/specificity, doable with a MLM)
  b. Tag heavy/light separately (potentially, expand this to train on IGH/IGK/IGL) to uniquely cater (special token) 
  b. Create train/val/test split on this + train appropriately 
  c. Purely to learn antibody sequence regularities, pairing grammar, etc
3. After training, learn to combine VH/VL together (concatenate) with the smaller (150k) component of the OAS to retrain the encoder
  a. Antigen encoder training (using new dataset)
  b. Antigen sequence encoder, cross-attention fusion, pooled joint embedding 
  c. This begins to inform the binding in a more generalized format
4. Represent antigen as the following property vector:
   Antibody Representation
   Antigen/Epitope representation (potentially structure…?)
   Assay Context
5. Add multitask heads (separate supervised property heads, then carefully tune it together) on different important components
  a. Attach these to important biophysical-aware concepts to “learn biology.” Fine-tune smaller heads for: 
     - Binder vs Non-binder
     - pKd/ΔG regression
     - Propensity for mutation?
6. Train a SAE on this fused antibody-antigen representation
  a. Collect activation info, train a sparse dictionary, then annotate the features (sparse-autoencoder-derived concept layer)
7. Constrained generation / lead generation of existing binder
  a. Focused on HCDR3, then expand slowly


