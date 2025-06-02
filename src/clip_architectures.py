import torch 
from abc import ABC, abstractmethod
import open_clip
from PIL import Image

"""
This code is adapted from the Capincho project (https://github.com/Andersonsr/capincho) by Anderson da Rosa.
"""

class Model(ABC):
    def __init__(self, device):
        self.backbone = None
        self.vision_preprocess = None
        self.language_preprocess = None
        self.device = device

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def visual_embedding(self, image_path):
        pass

    @abstractmethod
    def language_embedding(self, text):
        pass

    def similarity(self, text_features, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).max()

class OpenCLIP(Model):
    def visual_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.vision_preprocess(image).unsqueeze(0).to(self.device)
        return self.backbone.encode_image(image)

    def language_embedding(self, text):
        text = self.language_preprocess(text)
        return self.backbone.encode_text(text.to(self.device))

    def load_model(self):
        self.backbone, _, self.vision_preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=self.device
        )
        self.language_preprocess = open_clip.get_tokenizer('ViT-L-14')

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpenCLIP(device)
    model.load_model()
    image_path = "test.jpg"
    text = "this is a huge playground with basketball courts and tennis courts planned orderly inside"
    image_features = model.visual_embedding(image_path)
    text_features = model.language_embedding(text)
    similarity_score = model.similarity(text_features, image_features)
    print(f"Similarity score: {similarity_score.item()}")
    print("Model loaded and tested successfully.")
