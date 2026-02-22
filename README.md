# An autoencoder-based denoising filter for financial time-serie data
## Introduction

We create synthetic training data as noisy input for the autoencoder, which are random walks drawn from normal distribution, and its (exponential) moving averanges (periods of 2 to 50) as filtered/pure output. The motivation of denoising and using moving averange as training data is presented in the paper of 'Denoised Labels For Financial Time-series Data Via Self-supervised Learning'. The encoder structure we propose here, however, is a two 2D-Conv neural network.
For testing purposes we use the bitcoin hourly closing values retrieved from yahoo.

![](https://i.ibb.co/FkTrYJVs/1.png)

## Refernces:
1. https://doi.org/10.48550/arXiv.2112.10139
2. https://fcichos.github.io/website/notebooks/L14/2_AutoEncoder.html

## Disclaimer: this is not a financial advice and use at your own risk.
