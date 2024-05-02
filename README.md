This repository aims to record the related work on privacy-preserving graph learning visualization.

1. Learning Privacy-Preserving Graph Convolutional Network with Partially Observed Sensitive Attributes (WWW'2022)[[Paper]](https://dl.acm.org/doi/10.1145/3485447.3511975).
2. Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs (CIKM'2023)[[Paper]](https://arxiv.org/pdf/2308.13513.pdf).

### Run the code

```
Step 1: Train a GNN model on a dataset: python train_main.py  --dataset $DATASET.

Step 2: Install GNNLens2.

Step 3: Explain the trained model by using GNNExplainer: python explain_main.py --dataset $DATASET. 

Step 4: Launch GNNLens2 for visualization: gnnlens --logdir path+filename.
```
![](https://github.com/HuiHu1/Graph-Visualization/blob/main/Cora.JPG)

### Citation
```
@inproceedings{hu2022learning,
  title={Learning Privacy-Preserving Graph Convolutional Network with Partially Observed Sensitive Attributes},
  author={Hu, Hui and Cheng, Lu and Vap, Jayden Parker and Borowczak, Mike},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={3552--3561},
  year={2022}
}

@inproceedings{zhao2023unveiling,
  title={Unveiling the Role of Message Passing in Dual-Privacy Preservation on GNNs},
  author={Zhao, Tianyi and Hu, Hui and Cheng, Lu},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={3474--3483},
  year={2023}
}

```
