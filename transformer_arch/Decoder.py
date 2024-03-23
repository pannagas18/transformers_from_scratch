
import torch
from transformer_arch.PositionalEmbeddings import PositionalEmbeddings
from transformer_arch.ScaledDotProductAttention import ScaledDotProductAttention
from transformer_arch.PointwiseFeedForwardNetwork import PointwiseFeedForwardNetwork

class Decoder(torch.nn.Module):
  def __init__(self, config):
    super(Decoder, self).__init__()

    self.d_model = config.d_model
    self.n_layers = config.n_layers
    self.device = config.DEVICE

    self.pos_embeddings = PositionalEmbeddings(config) # LOOK INTO VOCAB, SHOULD IT BE JUST THE LEN OF VOCAB OR VOCAB ITSELF
    self.self_attention = ScaledDotProductAttention(config)
    self.encoder_decoder_attention = ScaledDotProductAttention(config)
    self.ffn = PointwiseFeedForwardNetwork(config)
    self.layer_norm = torch.nn.LayerNorm(self.d_model, device=self.device)

  def forward_once(self, x, encoder_output, encoder_attn_mask, decoder_attn_mask):
    """one decoder layer"""
    # print("self attention decoder")
    self_attention_output = self.self_attention(x, decoder=True, mask=decoder_attn_mask) # ADD CAUSAL MASKING
    output_layer_norm1 = self.layer_norm(self_attention_output + x)

    # print("encoder decoder attention")
    encoder_decoder_attention_output = self.encoder_decoder_attention(output_layer_norm1, encoder_output=encoder_output, decoder=True, mask=encoder_attn_mask) # ADD PAD MASKING
    output_layer_norm2 = self.layer_norm(encoder_decoder_attention_output + output_layer_norm1)

    ffn_output = self.ffn(encoder_decoder_attention_output)
    output_decoder_layer = self.layer_norm(ffn_output + output_layer_norm2)

    return output_decoder_layer

  def forward(self, decoder_input_ids, encoder_output, encoder_attn_mask, decoder_attn_mask):
    x = self.pos_embeddings(decoder_input_ids, decoder=True)
    for i in range(self.n_layers):
      x = self.forward_once(x, encoder_output, encoder_attn_mask, decoder_attn_mask)

    return x
  