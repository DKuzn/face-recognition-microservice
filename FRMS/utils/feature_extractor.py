from facenet_pytorch import InceptionResnetV1
import torch


class FeatureExtractor:
    def __init__(self):
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self._device)

    def extract_features(self, face: torch.Tensor) -> torch.Tensor:
        face = face.to(self._device)
        features = self._resnet(face.unsqueeze(0)).detach().cpu()
        return features[0]
