# DCN/DCNv2 Ranking Notes

## Overview
Deep & Cross Network (DCN) and its improved variant DCN V2 (DCNv2) are popular architectures for learning feature interactions in ranking and recommendation systems. They combine a cross network capturing explicit feature crosses with a deep network capturing implicit interactions. DCNv2 extends the original DCN by using full matrix weights to enable deeper and more expressive cross layers, allowing stacking multiple cross layers for higher-order interactions and better feature selectivity. DCN‑V2 has been shown to be more expressive and cost‑efficient while outperforming state‑of‑the‑art algorithms on real‑world ranking tasks.

## DCN vs DCNv2

| Aspect | DCN | DCNv2 |
| --- | --- | --- |
| Architecture | Combines a cross network (explicit feature crosses) and a deep network (implicit interactions). Each cross layer multiplies the input vector with learned weights and adds a bias. | Uses a modified cross network with full weight matrices, enabling more selective feature interactions. Cross layers can be stacked deeply to capture high‑order interactions. The cross network output is concatenated with the deep network output for prediction. |
| Expressiveness | Limited by using element‑wise products; the cross network may struggle with complex feature interactions. | More expressive due to full weight matrix and deeper stacking of cross layers, leading to better performance on benchmark ranking tasks. |
| Efficiency | Lower memory footprint and computation. | Slightly higher computational cost but still efficient; DCNv2 is designed to be cost‑effective for web‑scale ranking systems. |

## Loss Functions for Ranking

Ranking models can be trained with different loss functions depending on the desired objective:

- **Pointwise losses:** Treat ranking as a regression or classification problem by predicting scores for individual items (e.g., mean squared error or logistic loss). Simple and easy to optimize but may not directly optimize ranking metrics.

- **Pairwise losses:** Optimize the relative ordering between pairs of items (e.g., hinge loss or pairwise logistic loss). Pairwise losses encourage the model to rank positive items higher than negatives and are scalable to large datasets.

- **Listwise losses:** Directly optimize list‑level ranking metrics such as Normalized Discounted Cumulative Gain (NDCG) or Mean Average Precision (MAP). Listwise methods often yield better ranking performance but are more complex and computationally intensive.

## Training Tricks and Tips

When training DCN/DCNv2 for ranking tasks, consider these common strategies:

- **Feature engineering:** Use high‑dimensional sparse features (categorical embeddings) and continuous features. Normalize numeric inputs and carefully choose embedding sizes.

- **Cross network depth and selectivity:** In DCNv2, experiment with the number of cross layers. Stacking more layers can capture higher‑order interactions, but monitor for overfitting. Use weight regularization to avoid over‑complexity.

- **Loss function choice:** Pairwise or listwise losses often provide better ranking performance than simple pointwise losses. Listwise loss like NDCG can align training with evaluation metrics.

- **Training efficiency:** For large-scale data, mini-batch training with asynchronous data loading and optimized embeddings can improve throughput. Use learning-rate schedules, gradient clipping, and dropout to stabilize training.

- **Online/offline evaluation:** Evaluate ranking metrics (NDCG, MAP) on validation data. In production, A/B test different architectures and hyperparameters to ensure gains transfer to real users.

## Sample Code and Data

- `data/sample_ranking_data.csv` contains a tiny synthetic ranking dataset with continuous labels.
- `data/sample_esmm_data.csv` provides toy features plus click/conversion labels for multi-task ESMM experiments.
- `src/dcnv2_example.py` loads the ranking sample data and trains a DCNv2 model with full-matrix cross layers. Run `python src/dcnv2_example.py` to see a self-contained example.
- `src/esmm_example.py` implements the ESMM paper’s shared-bottom CTR/CTCVR setup. Execute `python src/esmm_example.py` to train on the synthetic click/conversion data.
- Prefer PyTorch? Use `src/dcnv2_torch_example.py` and `src/esmm_torch_example.py`, which mirror the TensorFlow demos with native Torch modules.
- `notebooks/dcnv2_sample.ipynb` mirrors the DCNv2 script inside a Jupyter workflow so you can tweak the architecture or inspect intermediate tensors interactively.
