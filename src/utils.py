from tqdm import tqdm
import torch
import open_clip
import clip
import random
import numpy as np
"""
This code is adapted from CLIP-LoRA (https://github.com/MaxZanella/CLIP-LoRA) by Max Zanella.
"""

def get_zero_shot_classifier(classnames, template, model):
    """
    Get the zero-shot classifier weights for the given class names and template.
    
    Args:
        classnames (list): List of class names.
        template (list): List of templates for text prompts.
        model: The model to use for feature extraction.
        
    Returns:
        torch.Tensor: The classifier weights.
    """
    with torch.no_grad():
        weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            weights.append(class_embedding)
        weights = torch.stack(weights, dim=1).cuda()
    return weights

# def pre_load_features(model, loader):
#     """
#     Preload features from the dataset using the provided model.
    
#     Args:
#         model: The model to use for feature extraction.
#         loader: DataLoader containing the dataset.
        
#     Returns:
#         A tuple of (features, labels).
#     """
#     features, labels = [], []
#     with torch.no_grad():
#         for images, target in tqdm(loader):
#             images, target = images.cuda(), target.cuda()
#             image_features = model.visual_embedding(images)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             features.append(image_features.cpu())
#             labels.append(target.cpu())
#         features, labels = torch.cat(features), torch.cat(labels)
    
#     return features, labels

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc

def debug_similarity(cosine_similarity, target, classnames, epoch_num, batch_idx):
    """
    Prints a debug summary of the cosine similarities for the first item in a batch.
    """
    # Only print for the first batch of each epoch to avoid too much output
    if batch_idx != 0:
        return

    print(f"\n--- Debugging Cosine Similarities (Epoch {epoch_num}, Batch 0) ---")
    
    # Get the model's predictions for the whole batch
    preds = torch.argmax(cosine_similarity, dim=1)
    
    # Let's inspect the very first example in the batch
    idx_to_inspect = 0
    
    true_label_idx = target[idx_to_inspect].item()
    pred_label_idx = preds[idx_to_inspect].item()
    
    true_classname = classnames[true_label_idx]
    pred_classname = classnames[pred_label_idx]
    
    # Get the similarity score that the model assigned to the TRUE class
    score_for_true_class = cosine_similarity[idx_to_inspect, true_label_idx].item()
    
    # Get the HIGHEST similarity score (the one the model chose as its prediction)
    score_for_pred_class = cosine_similarity[idx_to_inspect, pred_label_idx].item()

    print(f"Example #{idx_to_inspect}:")
    print(f"  - Ground Truth:           '{true_classname}' (label: {true_label_idx})")
    print(f"  - Model's Prediction:       '{pred_classname}' (label: {pred_label_idx})")
    print(f"  - Score for Correct Class:  {score_for_true_class:.4f}")
    print(f"  - Highest Score (Predicted): {score_for_pred_class:.4f}")
    
    if true_label_idx == pred_label_idx:
        print("  - Result: CORRECT")
    else:
        print("  - Result: INCORRECT")
    print("----------------------------------------------------------\n")

def save_results(args, test_accuracy, last_train_loss):
    """
    Saves the key hyperparameters and results of a training run to a text file.
    Appends to the file if it already exists.
    """
    output_file = "testing_hyperparameters.txt"
    
    # The list of important arguments you want to save
    important_args = [
        "shots", "position", "encoder", "params", "dropout_rate", "lr", 
        "n_iters", "alpha", "num_blocks", "block_rank"
    ]
    
    # Use 'a' mode to append to the file. It will be created if it doesn't exist.
    with open(output_file, 'a') as f:
        f.write("--- New Experiment Run ---\n")
        
        # Write the hyperparameters
        f.write("Hyperparameters:\n")
        for arg_name in important_args:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                f.write(f"  - {arg_name}: {value}\n")
        
        # Write the final results
        f.write("Results:\n")
        f.write(f"  - Final Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"  - Last Training Loss: {last_train_loss:.4f}\n")
        
        f.write("--------------------------\n\n")

    print(f"Results of this run were saved to {output_file}")

def set_random_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}")