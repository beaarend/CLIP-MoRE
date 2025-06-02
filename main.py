import torch
import argparse 
from src.clip_architectures import OpenCLIP


device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    parser = argparse.ArgumentParser(description="OpenCLIP Model Loader")
    parser.add_argument("--model", type=str, default="open_clip", help="Model to load")
    parser.add_argument("--backbone", type=str, default="ViT-B-32", help="Backbone model to use")

    args = parser.parse_args()

    if args.model != "open_clip" or args.backbone != "ViT-B-32":
        raise ValueError("Currently, only 'OpenCLIP' model with 'ViT-B-32' backbone is supported.")
    
    model = OpenCLIP(device)
    model.load_model()

    # create datasets and loaders
    # run more

if __name__ == "__main__":
    main()