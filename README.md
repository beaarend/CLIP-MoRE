# CLIP-MoRE

Applying **MoRE (Monarch Structured Fine-Tuning)** to CLIP.

---

## Background

- [MoRE (Monarch Structured Fine-Tuning)](https://github.com/SprocketLab/sparse_matrix_fine_tuning) — core algorithm for structured fine-tuning with Monarch matrices.
- [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA) — used as reference for adapting LoRA layers into CLIP.
- Base model: `ViT-B/32` trained on `laion2b_s34b_b79k` from OpenCLIP.

We chose **OpenCLIP** instead of original CLIP for better mixed precision support.

---

## Installation

### Create environment

```bash
conda create -y --name CLIP-MoRE python=3.10.0
conda activate CLIP-MoRE
```
### Install dependencies
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

### Run program
```bash
python3 main.py <args not defined>
```

## Steps
- [ ] Create base layer (LoRALayer to StructuredLinear)
- [ ] Create linear layer (LinearLoRA to MonarchLinear)
- [ ] MultiHeadAttention module uses linear layer, so as long as the linear layer is done, so is the MHA module
- [ ] Replace layers using apply_monarch_layers
- [ ] Test only using OxfordFlowers for now

### Thoughts

I need to create blockdiagonal1 and blkdiag2 to actually calculate the monarch matrixes
output = dense.forward(x) + monarch_linear.forward(x)
in_features & out_features = mha.embed_dim || => vit-b-32 = 768

are we able to use triton kernel? I THINK SO?
n_blocks = 4 in the original paper > Empirically, performance drops drastically when N>4
r_blk = 2?
