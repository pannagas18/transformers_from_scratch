
import torch

class PointwiseFeedForwardNetwork(torch.nn.Module):
  def __init__(self, config):
    super(PointwiseFeedForwardNetwork, self).__init__()

    self.d_model = config.d_model
    self.hidden_dim = config.hidden_dim
    self.device = config.DEVICE

    self.linear1 = torch.nn.Linear(in_features=self.d_model, out_features=self.hidden_dim, bias=True, device=self.device)
    self.relu = torch.nn.ReLU()
    self.linear2 = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.d_model, bias=True, device=self.device)

  def forward(self, x):
    x = self.relu(self.linear1(x))
    x = self.linear2(x)
    return x
