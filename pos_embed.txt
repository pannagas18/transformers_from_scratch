import torch
batch = 32
d_model = 512
seq = 512
vocab = 5000
print(torch.version)
pos = torch.arange(0,seq)
i = torch.arange(0,d_model/2)[:d_model]

embedding_layer = torch.nn.Embedding(vocab, d_model)
indices = torch.arange(0,seq).repeat(batch, 1) # this should be the batched seq token indices
token_embeddings = embedding_layer(indices)
#token_embeddings.shape

positional_embeddings = torch.zeros([seq, d_model])
for p in pos:
  div = p/torch.pow(10000, ((2*i)/d_model))
  sine_embed = torch.sin(div)
  cosine_embed = torch.cos(div)
  # interleave = torch.stack((sine_embed,cosine_embed), dim=1).view(-1)[:d_model]
  positional_embeddings[p] = torch.stack((sine_embed,cosine_embed), dim=1).view(-1)[:d_model]
#positional_embeddings.shape

print((token_embeddings + positional_embeddings).shape) # B,S,D)
