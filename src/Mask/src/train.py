import torch
import torchvision

from src.dataset import WildParkMaskDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device.")

dataset = WildParkMaskDataset(img_path="./data/frames/train/", ann_path="./data/annotations/train.json", transforms=None)

# model = model.get_model_instance_segmentation(3, pretrained=True).to(device)
# model.eval()

# x = [torch.rand(3, 300, 400).to(device)]
# pred = model(x)
# print(pred[0]['masks'].shape)
