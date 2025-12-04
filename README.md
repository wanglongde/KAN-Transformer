# KAN-Transformer
A minimal implementation exploring Kolmogorov-Arnold Networks as alternatives for MLP feed-forward layers in Transformer-based language models. This project includes training and evaluation scripts for character-level language modeling on the Tiny Shakespeare dataset, with experiments on width, grid size, depth, and embedding dimension scaling. The implementation is based on the architecture presented in the paper *"KAN-Transformers: Efficient Language Modeling with Kolmogorov–Arnold Networks"*.

## Architecture

The standard Transformer FFN layer

$$
\text{FFN}_{\text{MLP}}(\mathbf{x}) = \mathbf{W}_2(\sigma(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)) + \mathbf{b}_2
$$

is replaced with a KAN-based layer

$$
\text{FFN}_{\text{KAN}}(\mathbf{x}) = \Phi_2(\Phi_1(\mathbf{x})),
$$

where each KAN linear transformation \(\Phi\) implements learnable activation functions using B-splines, i.e.,

$$
\phi_{j,i}(x_i) = w^{\text{base}}_{j,i}\cdot\text{SiLU}(x_i) + \sum_{k=0}^{G-1} c_{j,i,k}\cdot\mathcal{B}_k(x_i).
$$

Here \(G\) is the grid size, `$\mathcal{B}_k$` are B-spline basis functions, and `$c_{j,i,k}$` are learnable spline coefficients.

## Acknowledgments

This implementation uses the pykan library by Liu et al. (2024) for KAN layers. The KAN formulation follows the original work: "KAN: Kolmogorov–Arnold Networks" by Ziming Liu et al.

## Quick Start
```bash
pip install -r requirements.txt
python train.py
