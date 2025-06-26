import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from morelib.utils import apply_monarch, mark_only_monarch_as_trainable, get_monarch_parameters, save_monarch, load_monarch
from src.utils import cls_acc, get_zero_shot_classifier, debug_similarity, save_results
import clip
import random
from PIL import Image, ImageDraw, ImageFont
import os
import torchvision.transforms.functional as TF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VALIDATION = False

"""
Part of this code is adapted from CLIP-LoRA (https://github.com/MaxZanella/CLIP-LoRA) by Max Zanella.
"""

def denormalize_tensor_to_pil(tensor):
    """
    Converts a normalized PyTorch tensor back to a PIL Image.
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    denorm_tensor = tensor.clone()
    
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    pil_image = TF.to_pil_image(denorm_tensor)
    return pil_image


def evaluate_classifier(args, model, test_loader, dataset, preprocess):
    list_monarch_layers = apply_monarch(args, model)
    load_monarch(args, list_monarch_layers)
    
    save_dir = "image_results/"
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        template = dataset.template[0]
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            tokenized_texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(tokenized_texts)
        text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    all_test_items = []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Collecting"):
            for i in range(images.size(0)):
                all_test_items.append((images[i].cpu(), targets[i].cpu()))
    
    num_to_sample = 15
    if len(all_test_items) < num_to_sample:
        print(f"Warning: Only {len(all_test_items)} images available. Using all of them.")
        num_to_sample = len(all_test_items)
    
    sampled_items = random.sample(all_test_items, num_to_sample)

    for i, (image_tensor, target) in enumerate(tqdm(sampled_items, desc="Processing Random Images")):

            target = target - 1 
            image_gpu_tensor = image_tensor.to(device).unsqueeze(0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = model.encode_image(image_gpu_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            _, pred_index_tensor = cosine_similarity.topk(1, dim=1)
            
            gt_index = target.item()
            pred_index = pred_index_tensor[0].item()

            ground_truth_name = dataset.classnames[gt_index]
            predicted_name = dataset.classnames[pred_index]

            annotated_image = denormalize_tensor_to_pil(image_tensor)

            draw = ImageDraw.Draw(annotated_image)
            try:
                font = ImageFont.truetype("arial.ttf", size=20)
            except IOError:
                font = ImageFont.load_default()

            gt_text = f"Ground Truth: {ground_truth_name.replace('_', ' ')}"
            pred_text = f"Prediction: {predicted_name.replace('_', ' ')}"
            is_correct = (ground_truth_name == predicted_name)
            text_color = "green" if is_correct else "red"

            draw.rectangle((2, 2, 300, 50), fill=(0, 0, 0, 128))
            draw.text((5, 5), gt_text, font=font, fill="white")
            draw.text((5, 27), pred_text, font=font, fill=text_color)

            safe_gt = ground_truth_name.replace('_', '-').replace(' ', '')
            safe_pred = predicted_name.replace('_', '-').replace(' ', '')
            filename = f"{i + 1:02d}_GT_{safe_gt}_PRED_{safe_pred}.png"
            annotated_image.save(os.path.join(save_dir, filename))

    print(f"\nFinished! Saved {len(sampled_items)} random annotated images to the '{save_dir}' directory.")

def evaluate_monarch(clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            target = target - 1
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc

def check_parameters(model, flag):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if trainable_params == 0:
        print(f"Warning: No trainable parameters found in the model after {flag} MoRE application.")
    if total_params == 0:
        print(f"Warning: No parameters found in the model after {flag} MoRE application.")

    print(f"{flag} Parameter Count ---")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params:     {total_params:,}")
    print(f"Percentage:       {100 * trainable_params / total_params:.4f}%")
    print(f"---------------------------------\n")

def run_monarch_training(args, model, logit_scale, dataset, train_loader, val_loader, test_loader):
    """
    Run the training loop for the model with MoRE layers.
    """
    print("Starting MoRE training...")

    # Get the text features for the zero-shot classifier
    textual_features = get_zero_shot_classifier(dataset.classnames, dataset.template, model)

    check_parameters(model, "Before MoRE")

    # Apply Monarch layers to the model
    list_monarch_layers = apply_monarch(args, model)
    model.cuda()

    if args.eval_only:
        print("Evaluation mode only. Skipping training.")
        acc = evaluate_monarch(model, test_loader, dataset)
        print(f"Validation accuracy: {acc:.4f}")
        # return

    mark_only_monarch_as_trainable(model, "monarch_only")

    check_parameters(model, "After MoRE")

    exit()

    total_iters = args.n_iters * args.shots
    warmup_iters = int(total_iters * 0.1)  

    optimizer = torch.optim.AdamW(get_monarch_parameters(model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_iters, 
        num_training_steps=total_iters
    )
    print("Optimizer and scheduler initialized.")

    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False
    epoch_num = 0
    while count_iters < total_iters:
        model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            target = target-1
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity.to(torch.float32), target)
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(get_monarch_parameters(model, bias="monarch_only"), 1.0)
            # debug_similarity(cosine_similarity, target, dataset.classnames, i, epoch_num)
            # if count_iters < 10:
            #     print("\n--- Checking Gradients ---")
            #     for name, param in model.named_parameters():
            #         if param.requires_grad:
            #             print(f"Gradient for {name}: {param.grad.mean().item() if param.grad is not None else 'None'}")
            #     print("--------------------------\n")

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
        
        epoch_num += 1
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            current_lr = scheduler.get_last_lr()[0]
            print('LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}'.format(current_lr, acc_train, loss_epoch))

        # Eval
        if VALIDATION:
            acc_val = evaluate_monarch(model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))
        
    
    acc_test = evaluate_monarch(model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))
    save_results(args, acc_test, loss_epoch)
    
    if args.save_path != None:
        save_monarch(args, list_monarch_layers)
    return
