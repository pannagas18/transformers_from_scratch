
import os
import torch
from Tokenizer import Tokenizer
from transformer_arch.Transformer import Transformer
from Loading import Loading
from configparser import ConfigParser

config_parser = ConfigParser()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "model_weights/ckpt_epoch3_steps3000"
CONFIG_SAVE_PATH = "model_weights/tfs_model0_config.ini"

# printing options
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class InferenceConfig:
  def __init__(self, batch, d_model, vocab, n_heads, hidden_dim, n_layers, DEVICE):
    self.batch = batch
    self.d_model = d_model
    self.vocab = vocab
    self.n_heads = n_heads
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.DEVICE = DEVICE

def inference():

    vocab = Tokenizer().get_vocab_size() # will be added depending on the vocab of the chosen tokenizer

    config_parser.read(CONFIG_SAVE_PATH)

    batch = 1 # for inference batch size will be 1
    d_model = config_parser.getint('model_configuration', 'd_model')
    n_heads = config_parser.getint('model_configuration', 'n_heads')
    hidden_dim = config_parser.getint('model_configuration', 'hidden_dim')
    n_layers = config_parser.getint('model_configuration', 'n_layers')

    config = InferenceConfig(batch, d_model, vocab, n_heads, hidden_dim, n_layers, DEVICE)
    # print(config.__dict__)

    # _input = "Opportunity, also known as MER-B or MER-1, is a robotic rover that was active on Mars from 2004 until 2018. Opportunity was operational on Mars for 5111 sols."
    _input = input("Enter the input article to be summarized:\n")

    os.system("clear" if os.name == 'posix' else 'cls')
    
    print(color.BOLD, color.YELLOW, "\nArticle:\n\n", color.END, color.YELLOW, _input, color.END)
    print(color.BOLD, color.GREEN, "\nSynopsis:\n", color.END, color.GREEN)
    
    loader = Loading()
    loader(done=False)
    
    tokenizer = Tokenizer().get_tokenizer()

    inference_encoder_input = tokenizer(_input, return_tensors="pt", padding=True)
    inference_decoder_input = {"input_ids":torch.tensor([0]), # 0 because <s>
                            "attention_mask":torch.tensor([1])}

    inference_encoder_input = {k:v.to(DEVICE) for k,v in inference_encoder_input.items()}
    inference_decoder_input = {k:v.to(DEVICE) for k,v in inference_decoder_input.items()}
    
    transformer_inference = Transformer(config)
    transformer_inference.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(DEVICE)))
    transformer_inference.eval()

    inference_preds = None

    while inference_preds != 2: # </s>
        with torch.inference_mode():

            inference_output_logits = transformer_inference(inference_encoder_input, inference_decoder_input)
            inference_preds = inference_output_logits.argmax(-1)[:,-1] # take the last predicted token

            inference_decoder_input = {"input_ids":torch.cat((inference_decoder_input["input_ids"], inference_preds), dim=-1),
                                    "attention_mask":torch.ones(inference_decoder_input["input_ids"].shape[0]+1)}

            generated_sequence = tokenizer.decode(inference_decoder_input["input_ids"], skip_special_tokens=False)

            # sys.stdout.write("\r" + generated_sequence)
            # sys.stdout.flush()

            # print(generated_sequence, end="\r")
    
    loader(done=True)
    print(generated_sequence)
    
    print("\n")

if __name__ == "__main__":
    inference()
