# Lab 2: A Neural Network from Scratch

## Dataset and preprocessing
- scikit-learn Digits dataset; 64 inputs scaled by dividing by 16.
- Labels one-hot encoded; stratified 70/30 split; random_state=42.

## Architecture
- 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax; W=16 and W=64.
- Weight shapes: (64,W), (W,W), (W,10); biases: (W,), (W,), (10,).
- Weights sampled from N(0,0.01), interpreted as variance 0.01 (standard deviation 0.1); biases initialized to zero.

## Forward pass and loss
- Output shape: (32, 10); row sums checked within 1e-6.
- Initial loss: 2.312799; ln(10)=2.302585.

## Gradient checking
- 10/10 selected weights passed relative error < 1e-6.
- Full details: results/gradient_check.csv.

## Training experiments
- Mini-batch SGD; batch size 32; 50 epochs; shuffled each epoch.
- Learning rates 0.001, 0.01, 0.1; hidden widths 16 and 64.
| Width | Learning rate | Final train loss | Test accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.290785 | 0.1315 |
| 16 | 0.01 | 1.325527 | 0.6426 |
| 16 | 0.1 | 0.080502 | 0.9389 |
| 64 | 0.001 | 2.188460 | 0.3870 |
| 64 | 0.01 | 0.306140 | 0.8981 |
| 64 | 0.1 | 0.018447 | 0.9593 |

Loss plot: `results/loss_curves.png`.

## Timing and analysis
| Width | Batch size | Forward median (ms) | Backward median (ms) |
|---:|---:|---:|---:|
| 16 | 32 | 0.065850 | 0.073350 |
| 64 | 32 | 0.100100 | 0.129350 |

- Timings are medians over 100 runs for each hidden width.
- Both passes use matrix multiplications. Forward propagation computes activations and logits. Backpropagation computes weight gradients and propagates error signals through the layers. Compare the measured medians above to identify which step dominated in this run.

## Implementation
- Network, softmax, cross-entropy, gradients, and training loop implemented with NumPy; no PyTorch or TensorFlow used.
