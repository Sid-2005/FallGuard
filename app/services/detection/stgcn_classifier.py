import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class STGCN_Mock(nn.Module):
    """
    A placeholder for the ST-GCN++ architecture.
    In a real scenario, this would include the Spatial-Temporal Graph Convolution layers
    (e.g., from the reference repos like punpayut/Fall-Detection).
    """
    def __init__(self, num_classes=2, in_channels=2, num_nodes=17, edge_importance_weighting=True):
        super().__init__()
        self.num_classes = num_classes
        # Minimal linear layers just to have some weights to load/test
        self.fc = nn.Linear(in_channels * num_nodes * 30, num_classes)
        
    def forward(self, x):
        # x shape: (N, C, T, V) => (Batch, Channels, Time, Vertices)
        N = x.size(0)
        x = x.view(N, -1)
        x = self.fc(x)
        return x

class STGCNClassifier:
    def __init__(self, model_path='stgcn_weights.pth', device='cpu'):
        self.device = torch.device(device)
        self.model = STGCN_Mock().to(self.device)
        self.weights_loaded = False
        
        if os.path.exists(model_path):
            try:
                # Handle PyTorch 2.6 weights_only strictness
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.weights_loaded = True
                self.model.eval()
                logger.info(f"Loaded ST-GCN++ weights from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ST-GCN++ weights: {e}")
        else:
            logger.warning(f"ST-GCN++ weights not found at {model_path}. Using heuristic fallback for inference.")

    def predict(self, buffer_keypoints: list) -> float:
        """
        buffer_keypoints: list of length up to 30.
        Each element is a dictionary of keypoints from YOLO-Pose, e.g., { 'nose': (x, y), ... }
        Returns a fall probability between 0.0 and 1.0.
        """
        if not buffer_keypoints or len(buffer_keypoints) < 5:
            return 0.0

        if self.weights_loaded:
            # Prepare tensor shape (1, C=2, T=30, V=17)
            # Not fully implemented yet due to lack of weights
            return 0.0

        # Heuristic fallback (since we don't have real weights yet):
        # We look at the body angle of the last frame in the buffer
        last_kp = buffer_keypoints[-1]
        
        try:
            if 'shoulder_mid' in last_kp and 'hip_mid' in last_kp:
                sm = last_kp['shoulder_mid']
                hm = last_kp['hip_mid']
                dx = hm[0] - sm[0]
                dy = hm[1] - sm[1]
                angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx)+1e-6)))
                
                # If angle < 35 (near horizontal), we assume fall for this test
                if angle < 35:
                    return 0.85 # High probability fall
        except Exception:
            pass
            
        return 0.1 # Low probability
