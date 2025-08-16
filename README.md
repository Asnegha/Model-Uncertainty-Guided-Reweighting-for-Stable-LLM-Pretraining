# Model Uncertainty-Guided Reweighting for Stable LLM Pretraining

Large Language Models (LLMs) have achieved remarkable performance across a wide range of NLP benchmarks. However, their success heavily depends on the quality of the pretraining data. Web-scale datasets often contain noisy or low-quality samples that can slow convergence and reduce model accuracy.

This project introduces **MetaEMReweighting**, a novel uncertainty-aware training algorithm that improves LLM pretraining by reweighting training examples based on their predicted difficulty.

---

## Key Features

- **Uncertainty-Based Sample Weighting:** Adjusts the importance of each training sample using estimated uncertainty from a proxy model.
- **Gaussian Mixture Model (GMM) Categorization:** Classifies samples into *easy*, *hard*, and *noisy* based on loss statistics.
- **Dynamic Weight Modulation:** Applies category-based weights to enhance the learning dynamics of the main LLM.

---

## Method Overview

1. **Proxy Model Loss Estimation:**  
   A small proxy model computes the mean and variance of batch-level loss for each training sample.

2. **Sample Categorization via GMM:**  
   Samples are clustered into three categories — *easy*, *hard*, and *noisy* — using a Gaussian Mixture Model on loss statistics.

3. **Weight Assignment:**  
   Category-specific weights are applied to influence the training dynamics of a larger auto-regressive LLM.

---

## 📊 Results

The MetaEMReweighting method consistently improves few-shot performance in a 5-shot setting across five benchmarks:

| Benchmark       | Baseline | MetaEMReweighting |
|------------------|----------|-------------------|
| **LogiQA**        | —        | **+0.30**          | 
| **PiQA**          | —        | **+0.40**          | 
| **ARC-Challenge** | —        | **+0.30**          | 

These gains validate the effectiveness of uncertainty-aware reweighting in enhancing LLM training quality.



