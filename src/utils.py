from tqdm import tqdm
import torch
import open_clip

"""
This code is adapted from CLIP-LoRA (https://github.com/MaxZanella/CLIP-LoRA) by Max Zanella.
"""

# def get_zero_shot_classifier(classnames, template, model):
#     """
#     Get the zero-shot classifier weights for the given class names and template.
    
#     Args:
#         classnames (list): List of class names.
#         template (list): List of templates for text prompts.
#         model: The model to use for feature extraction.
        
#     Returns:
#         torch.Tensor: The classifier weights.
#     """
#     with torch.no_grad():
#         weights = []
#         for classname in classnames:
#             classname = classname.replace('_', ' ')
#             texts = [t.format(classname) for t in template]
#             texts = model.language_preprocess(texts).cuda()
#             class_embeddings = model.backbone.encode_text(texts)
#             class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#             class_embedding = class_embeddings.mean(dim=0)
#             class_embedding /= class_embedding.norm()
#             weights.append(class_embedding)
#         weights = torch.stack(weights, dim=1).cuda()
#     return weights

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