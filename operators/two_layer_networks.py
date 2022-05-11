import torch


class TwoLayerFCNet(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int):
        super(TwoLayerFCNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, latent_dim)
        self.fc2 = torch.nn.Linear(latent_dim, output_dim)
    
    def forward(self, X: torch.Tensor):
        return self.fc2(torch.nn.functional.relu(self.fc1(X)))