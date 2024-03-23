
import torch
import torch.nn.functional as F

class ScaledDotProductAttention(torch.nn.Module):
  def __init__(self, config):
    super(ScaledDotProductAttention, self).__init__()

    self.d_model = config.d_model
    self.n_heads = config.n_heads
    self.batch = config.batch
    self.device = config.DEVICE

    self.K_linear = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model, bias=False, device=self.device)
    self.Q_linear = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model, bias=False, device=self.device)
    self.V_linear = torch.nn.Linear(in_features = self.d_model, out_features = self.d_model, bias=False, device=self.device)

    assert self.d_model % self.n_heads == 0, f"The input embedding of dimension {self.d_model} cannot be equally divided by number of heads {self.n_heads}"
    self.d_k = self.d_model//self.n_heads

  def forward(self, input_embeddings, encoder_seq_L=None, encoder_output=None, decoder=False, mask=None):

    # OPTIMIZE THIS (DECODER ONLY AND ENCODER ONLY WILL BE USING THE SAME VARIABLES TO CALCULATE K AND V)
    if decoder:
      # decoder self-attention
      self.seq_L = decoder_seq_L = input_embeddings.shape[-2] # B,S,D
      K = self.K_linear(input_embeddings)
      V = self.V_linear(input_embeddings)

    if (decoder) and (encoder_output is not None):
      # encoder-decoder cross-attention
      K = self.K_linear(encoder_output)
      V = self.V_linear(encoder_output)
      self.seq_L = encoder_output.shape[-2] # B,S,D

    elif encoder_seq_L is not None:
      # encoder self-attention
      self.seq_L = encoder_seq_L
      K = self.K_linear(input_embeddings)
      V = self.V_linear(input_embeddings)

    Q = self.Q_linear(input_embeddings)

    K_heads = K.view(self.batch, self.seq_L, self.n_heads, self.d_k).permute(0,2,1,3) # B, n_heads, seq_L, d_k
    if encoder_output is not None:
      # encoder-decoder cross-attention
      Q_heads = Q.view(self.batch, decoder_seq_L, self.n_heads, self.d_k).permute(0,2,1,3)
    else:
      # encoder self-attention, decoder self-attention
      Q_heads = Q.view(self.batch, self.seq_L, self.n_heads, self.d_k).permute(0,2,1,3)
    V_heads = V.view(self.batch, self.seq_L, self.n_heads, self.d_k).permute(0,2,1,3)

    # add masking (causal and padding) to attn_mat
    attn_mat = Q_heads@(K_heads.mT)
    attn_mat /= ((self.d_k)**0.5)

    if mask is not None:
      if encoder_output is not None:
        # apply encoder-decoder masking (only padding)
        pad_mask = mask.view(self.batch, 1, 1, self.seq_L).eq(0)
        masked_attn_mat = torch.masked_fill(attn_mat, pad_mask, -torch.inf)
      elif (decoder) and (encoder_output is None):
        # apply decoder causal masking
        causal_out = torch.tril(attn_mat)
        masked_attn_mat = torch.where(causal_out==0., -torch.inf, causal_out)
      else:
        # apply encoder masking (padding mask)
        pad_mask = mask.repeat_interleave(self.seq_L,0).view(self.batch, 1, self.seq_L, self.seq_L).eq(0)
        masked_attn_mat = torch.masked_fill(attn_mat, pad_mask, -torch.inf)

    attn_scores = F.softmax(masked_attn_mat, dim=-1)
    output_heads = attn_scores @ V_heads

    if encoder_output is not None:
      # encoder-decoder cross-attention
      output = output_heads.permute(0,2,1,3).reshape(self.batch, decoder_seq_L, -1)
    else:
      # encoder self-attention, decoder self-attention
      output = output_heads.permute(0,2,1,3).reshape(self.batch, self.seq_L, -1)

    return output
