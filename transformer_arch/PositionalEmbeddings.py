
import torch

class PositionalEmbeddings(torch.nn.Module):
  def __init__(self, config):
    super(PositionalEmbeddings, self).__init__()

    self.device = config.DEVICE
    self.vocab = config.vocab
    self.d_model = config.d_model
    self.embedding_layer = torch.nn.Embedding(num_embeddings=self.vocab, embedding_dim=self.d_model, device=self.device)
    self.i = torch.arange(0,self.d_model/2)[:self.d_model]

  def get_positional_embeddings(self, seq_L=None):

    pos = torch.arange(0,seq_L)

    positional_embeddings = torch.zeros([seq_L, self.d_model]).to(self.device)

    for p in pos:
      div = p/torch.pow(10000, ((2*self.i)/self.d_model))
      sine_embed = torch.sin(div)
      cosine_embed = torch.cos(div)
      positional_embeddings[p] = torch.stack((sine_embed,cosine_embed), dim=1).view(-1)[:self.d_model]

    return positional_embeddings

  def forward(self, input_ids, encoder_seq_L=None, decoder=False): # input_ids => batched seq token indices

    token_embeddings = self.embedding_layer(input_ids)

    if decoder:
      positional_embeddings = self.get_positional_embeddings(seq_L=input_ids.shape[-1])
    elif encoder_seq_L is not None:
      positional_embeddings = self.get_positional_embeddings(seq_L=encoder_seq_L)

    return token_embeddings + positional_embeddings
