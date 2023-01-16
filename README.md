This repository aims to record the related work on graph visualization in machine learning.

GNNExplainer: Generating Explanations for Graph Neural Networks (NeurIPS 2019) [[Paper]](https://arxiv.org/pdf/1903.03894.pdf)[[Code]](https://github.com/RexYing/gnn-model-explainer).

### Run the code

```
Step 1: Train a GNN model on a dataset: python train_main.py  --dataset $DATASET.

Step 2: Install GNNLens2.

Step 3: Explain the trained model: python explain_main.py --dataset $DATASET.

Step 4: Launch GNNLens2 for visualization: gnnlens --logdir path+filename.
```
