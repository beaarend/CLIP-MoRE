import torch
import argparse
import torchvision.transforms as transforms 
from src.clip_architectures import OpenCLIP
from datasets import build_dataset
from datasets.utils import build_data_loader
from training import run_monarch_training

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    parser = argparse.ArgumentParser(description="OpenCLIP Model Loader")
    parser.add_argument("--model", type=str, default="open_clip", help="Model to load")
    parser.add_argument("--backbone", type=str, default="ViT-B/32", help="Backbone model to use")
    parser.add_argument("--dataset", type=str, default="oxford_flowers", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loader")
    parser.add_argument("--shots", type=int, default=4, help="Number of shots for few-shot learning")
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only without training', default=True)

    parser.add_argument("--num_blocks", type=int, default=16, help="Number of blocks in the model")
    parser.add_argument("--block_rank", type=int, default=16, help="Rank of the blocks in the model")

    args = parser.parse_args()

    if args.model != "open_clip" or args.backbone != "ViT-B/32":
        raise ValueError("Currently, only 'OpenCLIP' model with 'ViT-B/32' backbone is supported.")
    
    model = OpenCLIP(device)
    model.load_model()
    logit_scale = 100

    if args.dataset != "oxford_flowers":
        raise ValueError("Currently, only 'oxford_flowers' datasets are supported.")
    
    dataset = build_dataset(args.dataset, root_path="", shots=args.shots)
    val_loader = build_data_loader(dataset.val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = build_data_loader(dataset.test, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=4)

    run_monarch_training(args, model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    main()