# CLIP-MoRE
Applying MoRE to CLIP.

using open clip instead of clip because it allows using mixed precision and we dont have good machines
using vit-b-32 trained on laion2b_s34b_b79k

cria um env se nao tiver
```bash
conda create -y --name CLIP-MoRE python=3.10.0
conda activate CLIP-MoRE
```

```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

https://github.com/SprocketLab/sparse_matrix_fine_tuning -> structured linear here is equal to LoRALayer in LoRA repo

we need to add a layer that adapts nn.Linear and a new multiheadattention module 