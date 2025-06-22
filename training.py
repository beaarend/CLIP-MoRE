import torch
import torch.nn.functional as F
import tqdm
from morelib.utils import apply_monarch, mark_only_monarch_as_trainable, get_monarch_parameters
from src.utils import cls_acc, get_zero_shot_classifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_monarch(model, loader, text_features, device):
    model.backbone.eval()
    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc="Evaluating MoRE model")):
            images, target = images.to(device), target.to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Calculate logits by multiplying with the pre-computed text features
                cosine_similarity = image_features @ text_features

            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
            
    acc /= tot_samples
    return acc


def run_monarch_training(args, model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Run the training loop for the model with MoRE layers.
    """
    print("Starting MoRE training...")

    # Get the text features for the zero-shot classifier
    text_features = get_zero_shot_classifier(dataset.classnames, dataset.template, model)

    # Apply Monarch layers to the model
    list_lora_layers = apply_monarch(args, model)
    print(f"Applied {len(list_lora_layers)} MoRE layers to the model.")
    mark_only_monarch_as_trainable(model, bias='monarch_only')
    print("Marked only MoRE layers as trainable.")

    if args.eval_only:
        print("Evaluation mode only. Skipping training.")
        acc = evaluate_monarch(model, val_loader, text_features, device)
        print(f"Validation accuracy: {acc:.4f}")
        return

    total_iters = args.n_iters * args.shots
    optimizer = torch.optim.AdamW(get_monarch_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    print("Optimizer and scheduler initialized.")

    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
