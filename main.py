from time import sleep
import torch
import argparse
import torchvision.transforms as transforms 
from src.clip_architectures import OpenCLIP
from datasets import build_dataset
from datasets.utils import build_data_loader
from training import run_monarch_training
import clip
from src.utils import set_random_seed

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    parser = argparse.ArgumentParser(description="OpenCLIP Model Loader")
    parser.add_argument("--model", type=str, default="open_clip", help="Model to load")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="Backbone model to use")
    parser.add_argument("--dataset", type=str, default="oxford_flowers", help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loader")
    parser.add_argument("--shots", type=int, help="Number of shots for few-shot learning")
    parser.add_argument('--position', type=str, default='top1', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3', 'top1'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', help='list of attention matrices where putting a monarch') 
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the monarch module')
    parser.add_argument('--lr',  type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only without training', default=False)
    parser.add_argument('--save_path', type=str, default='results/', help='Path to save the trained model')
    parser.add_argument('--filename', default='lora_weights', help='file name to save the monarch weights (.pt extension will be added)')
    parser.add_argument('--alpha', type=float, help='scaling')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    parser.add_argument("--num_blocks", type=int, help="Number of blocks in the model")
    parser.add_argument("--block_rank", type=int, help="Rank of the blocks in the model")

    args = parser.parse_args()
    # set_random_seed(args.seed)

    # # print all the arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    sleep(1)

    # model = OpenCLIP(device)
    # model.load_model()
    # logit_scale = 50
    model, preprocess = clip.load(args.backbone)
    model.eval()
    logit_scale = model.logit_scale.exp().detach()

    if args.dataset != "oxford_flowers":
        raise ValueError("Currently, only 'oxford_flowers' datasets are supported.")
    
    dataset = build_dataset(args.dataset, root_path="", shots=args.shots)
    val_loader = build_data_loader(dataset.val, batch_size=args.batch_size, tfm=preprocess, is_train=False, shuffle=False, num_workers=8)
    test_loader = build_data_loader(dataset.test, batch_size=args.batch_size, tfm=preprocess, is_train=False, shuffle=False, num_workers=8)

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