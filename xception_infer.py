# xception_infer.py
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class XceptionLike(nn.Module):
    # Placeholder for a loaded model; replace with actual implementation or load via torch.hub/state_dict
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.out_features if hasattr(backbone, "out_features") else 2048, 1)

    def forward(self, x):
        feats = self.backbone(x) if hasattr(self.backbone, "__call__") else x
        logits = self.head(feats)
        return logits

class DeepfakeDetector:
    def __init__(self, weights_path: str, device: str = None, input_size: int = 299):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # Load your actual Xception/EfficientNet-based detector here
        # Example: load a scripted or state_dict model
        # self.model = torch.jit.load(weights_path, map_location=self.device)
        # For illustration, assume a simple scripted model with a sigmoid output
        self.model = torch.jit.load(weights_path, map_location=self.device)  # expects logits
        self.model.eval().to(self.device)

        # Typical Xception preprocessing for FF++:
        self.tf = T.Compose([
            T.ToTensor(),                 # HWC [0,255] -> CHW [0,1]
            T.Resize((self.input_size, self.input_size)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1,1] scale; adjust to your training
        ])

        self.sigmoid = nn.Sigmoid()

    @torch.no_grad()
    def predict_proba(self, img_rgb_np: np.ndarray) -> float:
        # img_rgb_np: HxWx3 uint8 RGB
        x = self.tf(img_rgb_np).unsqueeze(0).to(self.device)  # 1x3xHxW
        logits = self.model(x)  # 1x1 or 1xC where C=1
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = self.sigmoid(logits).flatten()[0].item()
        return float(prob)  # probability of "fake"
