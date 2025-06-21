import torch
import torch.nn.functional as F
import tqdm
from morelib.utils import apply_monarch

def run_monarch_training(args, model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Run the training loop for the model with MoRE layers.
    """
    print("Starting MoRE training...")

    # Apply Monarch layers to the model
    list_lora_layers = apply_monarch(args, model)
    print(f"Applied {len(list_lora_layers)} MoRE layers to the model.")

