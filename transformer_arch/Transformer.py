
import torch
import torch.nn.functional as F
from transformer_arch.Encoder import Encoder
from transformer_arch.Decoder import Decoder

class Transformer(torch.nn.Module):
  def __init__(self, config:dict):
    super(Transformer, self).__init__()

    self.d_model = config.d_model
    self.vocab = config.vocab
    self.device = config.DEVICE

    self.encoder = Encoder(config)
    self.decoder = Decoder(config)
    self.linear = torch.nn.Linear(in_features=self.d_model, out_features=self.vocab, bias=True, device=self.device)

  def forward(self, encoder_input, decoder_input):

    encoder_input_ids, decoder_input_ids = encoder_input["input_ids"], decoder_input["input_ids"]
    encoder_attn_mask, decoder_attn_mask = encoder_input["attention_mask"], decoder_input["attention_mask"]

    encoder_seq_L = encoder_input_ids.shape[-1]
    encoder_output = self.encoder(encoder_input_ids, encoder_attn_mask, encoder_seq_L)
    decoder_output = self.decoder(decoder_input_ids, encoder_output, encoder_attn_mask, decoder_attn_mask)
    output_probabilities = F.softmax(self.linear(decoder_output), dim=-1)

    return output_probabilities