# Self-Pruning Neural Network

This project implements a self-pruning neural network on the CIFAR-10 dataset using learnable gate parameters and L1 sparsity regularization. The model dynamically learns which weights are unnecessary and prunes them during training.

---

## 🚀 Approach

- Custom linear layer (`AdaptiveLinear`) with learnable gate parameters
- Gates are activated using the sigmoid function (values between 0 and 1)
- Effective weights = weight × gate values
- L1 regularization applied on gate values to encourage sparsity
- Trade-off between accuracy and sparsity controlled using λ (lambda)

---

## 🧠 Sparsity Mechanism

L1 regularization penalizes the sum of gate values.

Since gate values lie between 0 and 1, this pushes many of them toward 0.  
When a gate approaches 0, the corresponding weight becomes inactive → effectively pruned.

This allows the network to automatically remove less important connections and become sparse.

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 1e-5   | 55.29       | 11.77       |
| 1e-4   | 56.32       | 61.26       |
| 5e-4   | 56.78       | 94.05       |

---

## 📉 Gate Distribution

The distribution below shows how many weights are pruned (near 0) vs active.

![Gate Distribution](results/gate_distribution.png)

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python self_pruning_nn.py
