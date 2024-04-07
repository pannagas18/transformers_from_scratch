import torch
from torch.utils.data import DataLoader
from create_dataset import TransformerDataset
from Tokenizer import Tokenizer
from transformer_arch.Transformer import Transformer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
from pathlib import Path
import re
import argparse
from configparser import ConfigParser
config_parser = ConfigParser()

parser = argparse.ArgumentParser()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
  def __init__(self, batch, d_model, vocab, n_heads, hidden_dim, n_layers, DEVICE):
    self.batch = batch
    self.d_model = d_model
    self.vocab = vocab
    self.n_heads = n_heads
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.DEVICE = DEVICE

def train(epoch, transformer, encoder_dataloader, decoder_dataloader, optimizer, loss_fn, DEVICE, ckpt_model_name, writer):

    print("training...")
    transformer.train()
    avg_train_loss = 0
    train_iter = zip(encoder_dataloader, decoder_dataloader)

    with tqdm(train_iter, total=len(encoder_dataloader), unit="batch", position=0, leave=True) as tepoch:
        for batch_idx, (encoder_input, decoder_input) in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch}")

            encoder_input = {k:v.to(DEVICE) for k,v in encoder_input.items()}
            decoder_input = {k:v.to(DEVICE) for k,v in decoder_input.items()}


            target_ids = decoder_input["input_ids"][:,1:].to(DEVICE) # for all elements in the batch, don't consider <s>
            target_attn_mask = decoder_input["attention_mask"][:,1:].to(DEVICE) # for all elements in the batch, don't consider <s>
            target_padding_mask = target_attn_mask.eq(1)

            # remove the </s> (tokeninzer idx = 2) from the decoder input_ids
            decoder_input["input_ids"] = decoder_input["input_ids"][decoder_input["input_ids"]!=2].view(decoder_input["input_ids"].shape[0],-1) # b, decoder_seq_L

            optimizer.zero_grad()
            
            output_logits = transformer(encoder_input, decoder_input)

            # LOSS; calculate the loss with reduction = none and then multply by the padding mask
            loss_outputs = output_logits.permute(0,2,1)
            loss_with_pad = loss_fn(loss_outputs, target_ids) * target_padding_mask # reduction="none"
            loss = loss_with_pad[(target_padding_mask==1)].mean()
            
            loss.backward()
            optimizer.step()

            tepoch.set_postfix({"Loss": loss.item()})
            # tepoch.update(1)
            
            avg_train_loss += loss
            
            # todo; create checkpoints during training -> done
            if batch_idx%3500 == 0:
                # writer.add_scalar("Loss/train", avg_train_loss.item()/(batch_idx+1), epoch)
                torch.save(transformer.state_dict(), f"""training_ckpts/{ckpt_model_name}_ckpt_epoch{epoch}_steps{batch_idx}""")

    avg_train_loss /= len(encoder_dataloader)
    print("\nAVG TRAIN LOSS:", avg_train_loss.item(), "\n")

    writer.add_scalar("Loss/train", avg_train_loss, epoch)


def validation(epoch, transformer, encoder_dataloader, decoder_dataloader, loss_fn, best_validation_loss, DEVICE, MODEL_SAVE_PATH, writer):

    print("validating...")

    transformer.eval()
    avg_validation_loss = 0
    validation_iter = zip(encoder_dataloader, decoder_dataloader) # CHANGE THIS

    with torch.inference_mode():
        with tqdm(validation_iter, total=len(encoder_dataloader), unit="batch", position=0, leave=True) as tepoch:
            for batch_idx, (encoder_input, decoder_input) in enumerate(tepoch):

                encoder_input = {k:v.to(DEVICE) for k,v in encoder_input.items()}
                decoder_input = {k:v.to(DEVICE) for k,v in decoder_input.items()}

                target_ids = decoder_input["input_ids"][:,1:].to(DEVICE) # for all elements in the batch, don't consider <s>
                target_attn_mask = decoder_input["attention_mask"][:,1:].to(DEVICE) # for all elements in the batch, don't consider <s>
                target_padding_mask = target_attn_mask.eq(1)

                # remove the </s> (tokeninzer idx = 2) from the decoder input_ids
                decoder_input["input_ids"] = decoder_input["input_ids"][decoder_input["input_ids"]!=2].view(decoder_input["input_ids"].shape[0],-1) # b, decoder_seq_L
            
                output_logits = transformer(encoder_input, decoder_input)

                # LOSS; calculate the loss with reduction = none and then multply by the padding mask
                loss_outputs = output_logits.permute(0,2,1)
                loss_with_pad = loss_fn(loss_outputs, target_ids) * target_padding_mask # reduction="none"
                loss = loss_with_pad[(target_padding_mask==1)].mean()

                tepoch.set_postfix({"Loss": loss.item()})
                # tepoch.update(1)

                avg_validation_loss += loss

        avg_validation_loss /= len(encoder_dataloader)
        if avg_validation_loss < best_validation_loss:
            print("\n\nsaving best model...")
            torch.save(transformer.state_dict(), MODEL_SAVE_PATH+f"epoch_{epoch}")
            best_validation_loss = avg_validation_loss

        print("AVG VALIDATION LOSS:", avg_validation_loss.item())
        print("BEST VALIDATION LOSS:", best_validation_loss.item(), "\n")

        writer.add_scalar("Loss/test", avg_validation_loss, epoch)

    return best_validation_loss


def main():
    # CLI arguments, argparser
    parser.add_argument("--batch", help="batch size", type=int)
    parser.add_argument("--d_model", help="embedding dimension of the model", type=int)
    parser.add_argument("--n_heads", help="number of attention heads", type=int)
    parser.add_argument("--hidden_dim", help="hidden dimension of the model (linear layer)", type=int)
    parser.add_argument("--n_layers", help="number of encoder and decoder layers", type=int)
    parser.add_argument("--model_save_path", help="model save path", type=str)

    args = parser.parse_args()

    vocab = Tokenizer().get_vocab_size() # will be added depending on the vocab of the chosen tokenizer
    config = Config(args.batch, args.d_model, vocab, args.n_heads, args.hidden_dim, args.n_layers, DEVICE)
    # print(config.__dict__)
    
    folder_path = args.model_save_path.split('/')[0]
    Path("training_ckpts").mkdir(parents=True, exist_ok=True)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    time = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    match = re.search(r'/(\w+)$', args.model_save_path)
    # log dir for tensorboard visualization
    log_dir = f"logruns/{time}_{match.group(1)}" if match else f"logruns/{time}_model"
    writer = SummaryWriter(log_dir=log_dir)

    # save model config in order to load it during inference
    config_path = f"{args.model_save_path}_config.ini"
    config_parser.read(config_path)
    config_parser.add_section('model_configuration')
    config_parser.set('model_configuration', 'batch', f"{args.batch}")
    config_parser.set('model_configuration', 'd_model', f"{args.d_model}")
    config_parser.set('model_configuration', 'n_heads', f"{args.n_heads}")
    config_parser.set('model_configuration', 'hidden_dim', f"{args.hidden_dim}")
    config_parser.set('model_configuration', 'n_layers', f"{args.n_layers}")

    with open(config_path, 'w') as f:
        config_parser.write(f)

    dataset_name, dataset_version = "cnn_dailymail", "1.0.0"

    train_dataset_article, train_dataset_highlights, validation_dataset_article, validation_dataset_highlights, test_dataset_article, test_dataset_highlights = TransformerDataset(config=config,
                                                            dataset_name=dataset_name,
                                                            dataset_version=dataset_version).get_dataset()

    encoder_train_dataloader = DataLoader(train_dataset_article, batch_size=config.batch, drop_last=True)
    decoder_train_dataloader = DataLoader(train_dataset_highlights, batch_size=config.batch, drop_last=True)

    encoder_validation_dataloader = DataLoader(validation_dataset_article, batch_size=config.batch, drop_last=True)
    decoder_validation_dataloader = DataLoader(validation_dataset_highlights, batch_size=config.batch, drop_last=True)

    # encoder_test_dataloader = DataLoader(test_dataset_article, batch_size=config.batch, drop_last=True)
    # decoder_test_dataloader = DataLoader(test_dataset_highlights, batch_size=config.batch, drop_last=True)

    transformer = Transformer(config)
    # load model from a training checkpoint
    # transformer.load_state_dict(torch.load(r"training_ckpts\tfs_model0_ckpt_epoch1_steps14000", map_location=torch.device(DEVICE)))

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-4)

    EPOCH = 5
    best_validation_loss = float('inf') # validation loss
    for epoch in range(1, EPOCH+1):
        train(epoch, transformer, encoder_train_dataloader, decoder_train_dataloader, optimizer, loss_fn, DEVICE, f"{match.group(1)}" if match else "model", writer)
        best_validation_loss = validation(epoch, transformer, encoder_validation_dataloader, decoder_validation_dataloader, loss_fn, best_validation_loss, DEVICE, args.model_save_path, writer)
    writer.flush()

if __name__ == "__main__":
   main()
