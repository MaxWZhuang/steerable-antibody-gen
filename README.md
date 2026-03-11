# steerable-antibody-gen
Antibody Language Model, implementing steerability with SAE-derived concept labelling

Overall plan: 

1. Pretrain with PyTorch on antibody sequence biology first on OAS (Observed Antibody Space)
  a. This can be messier data (unpaired data for example, can generate VH and VL separately as long as they are labelled) – EXPLICITLY label VH vs VL   (separate LLMs??)
  b. Create train/val/test split on this, train appropriately
  c. Purely to learn antibody sequence regularities, pairing grammar, etc
2. After training, learn to combine VH/VL together (concatenate) with the smaller (300k) component of the OAS to retrain the encoder (transformer into vector)  
  a. Antigen encoder training (using new dataset)
  b. Antigen sequence encoder, cross-attention fusion, pooled joint embedding 
  c. This begins to inform the binding in a more generalized format
3. Represent antigen as the following property vector:
   Antibody Representation
   Antigen/Epitope representation (potentially structure…?)
   Assay Context
4. Add multitask heads (separate supervised property heads, then carefully tune it together) on different important components
  a. Attach these to important biophysical-aware concepts to “learn biology.” Fine-tune smaller heads for: 
     - Binder vs Non-binder
     - pKd/ΔG regression
     - Propensity for mutation?
5. Train a SAE on this fused antibody-antigen representation
  a. Collect activation info, train a sparse dictionary, then annotate the features (sparse-autoencoder-derived concept layer)
6. Constrained generation / lead generation of existing binder
  a. Focused on HDCR3, then expand slowly


