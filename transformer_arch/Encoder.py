
import torch
from transformer_arch.PositionalEmbeddings import PositionalEmbeddings
from transformer_arch.ScaledDotProductAttention import ScaledDotProductAttention
from transformer_arch.PointwiseFeedForwardNetwork import PointwiseFeedForwardNetwork

class Encoder(torch.nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()

    self.d_model = config.d_model
    self.n_layers = config.n_layers
    self.device = config.DEVICE

    self.pos_embeddings = PositionalEmbeddings(config) # LOOK INTO VOCAB, SHOULD IT BE JUST THE LEN OF VOCAB OR VOCAB ITSELF
    self.self_attention = ScaledDotProductAttention(config) # ADD PAD MASKING
    self.ffn = PointwiseFeedForwardNetwork(config)
    self.layer_norm = torch.nn.LayerNorm(self.d_model, device=self.device)

  def forward_once(self, x, attn_mask, encoder_seq_L):
    """one encoder layer"""
    # print("encoder self attention")
    self_attention_output = self.self_attention(x, encoder_seq_L=encoder_seq_L, mask=attn_mask)
    output_layer_norm1 = self.layer_norm(self_attention_output + x)

    ffn_output = self.ffn(output_layer_norm1)
    output_encoder_layer = self.layer_norm(ffn_output + output_layer_norm1)

    return output_encoder_layer

  def forward(self, encoder_input_ids, attn_mask, encoder_seq_L):
    x = self.pos_embeddings(encoder_input_ids, encoder_seq_L)
    for i in range(self.n_layers):
      x = self.forward_once(x, attn_mask, encoder_seq_L)

    return x
