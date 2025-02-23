import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
from torch.utils.data import DataLoader
import tiktoken
import time
import matplotlib.pyplot as plt
from gpt_download3 import download_and_load_gpt2
import ast


from flask import Flask, request, jsonify
app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


GPT_CONFIG_774M = {                                                #1558M parameters gpt
    "vocab_size": 50257,    # Vocabulary size
    "emb_dim": 1280,         # Embedding dimension                      #768 before this
    "n_heads": 20,          # Number of attention heads                 #12 before this
    "n_layers": 36,         # Number of layers                          #12 before this
    "drop_rate": 0.25,       # Dropout rate just changed to 0.3 from 0.1
    "qkv_bias": False       # Query-Key-Value bias
}

class TransformerBlock(nn.Module):                              #transformer architecture
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # 2*4*768
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
        # 2*4*768

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):                                                        #activation
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):                                                   #feedforward layer
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
        )

    def forward(self, x):
        return self.layers(x)
    
class MultiHeadAttention(nn.Module):                                                                #attention mechanism used
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    


class GPTModel(nn.Module):                                                      #model architecture
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_drop = nn.Dropout(cfg["drop_rate"])  # Transformer output dropout

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class EthDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["sentiment"]
        # Convert string to dictionary and extract the class

        label_dict = ast.literal_eval(label)  # Convert string to dictionary
        sentiment_class = label_dict["class"]  # Extract 'class' value
        if label_dict["class"] == "negative":
            sentiment_class = 0
        elif label_dict["class"] == "neutral":
            sentiment_class = 1
        else:
            sentiment_class = 2

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(sentiment_class, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    
def load_model(model_path, NEW_CONFIG):  # Helper function to load the saved model
    # Instantiate the model
    model = GPTModel(NEW_CONFIG)

    num_classes = 3
    model.out_head = torch.nn.Linear(in_features=NEW_CONFIG["emb_dim"], out_features=num_classes)  # Final output layer

    # Load the state dict with CPU mapping
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    for param in model.trf_blocks[-1].parameters():  # Experiment with this
        param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Set the model to evaluation mode
    model.eval()

    return model


def calc_accuracy_loader(data_loader, gpt, device, num_batches=None):
    gpt.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = gpt(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-large (774M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_774M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})



model_path = "Eth_predictions_10_epochs.pth"

# Load the model
gpt = load_model(model_path,NEW_CONFIG)
gpt.to(device)
tokenizer = tiktoken.get_encoding("gpt2")


# train_dataset = EthDataset(
#     csv_file="cryptonews_train.csv",
#     max_length=None,
#     tokenizer=tokenizer
# )

# print(train_dataset.max_length)
# val_dataset = EthDataset(
#     csv_file="cryptonews_val.csv",
#     max_length=train_dataset.max_length,
#     tokenizer=tokenizer
# )
# test_dataset = EthDataset(
#     csv_file="cryptonews_test.csv",
#     max_length=train_dataset.max_length,
#     tokenizer=tokenizer
# )

# print(test_dataset.max_length)




num_workers = 0
batch_size = 8

torch.manual_seed(123)

# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     drop_last=True,
# )

# val_loader = DataLoader(
#     dataset=val_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     drop_last=False,
# )

# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     drop_last=False,
# )

# print(f"{len(train_loader)} training batches")
# print(f"{len(val_loader)} validation batches")
# print(f"{len(test_loader)} test batches")
# # train_accuracy = calc_accuracy_loader(train_loader, gpt, device)
# # val_accuracy = calc_accuracy_loader(val_loader, gpt, device)
# # test_accuracy = calc_accuracy_loader(test_loader, gpt, device)

# # print("AFTER FINE TUNING")
# # print(f"Training accuracy: {train_accuracy*100:.2f}%")
# # print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# # print(f"Test accuracy: {test_accuracy*100:.2f}%")




"""END OF LOADING THE MODEL"""
"""START OF TESTING"""

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    print("Raw logits: ",logits)
    print("\nProbabilities:", probabilities)

    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    if predicted_label==0:
        return "going down"
    elif predicted_label==1:
        return "almost stable"
    else:
        return "going up"
    


@app.route('/receive_news', methods=['POST'])
def receive_news():
    data = request.json.get("news", [])

    print("\n🔹 All Fetched News Titles:\n")

    pos=0
    neg=0
    neu=0

    i=0
    for news in data:
        if i>15:
            break
        print(f"- {news['title']}")
        text = f"{news['title']}"
        classification = classify_review(text, gpt, tokenizer, device, max_length=1000)
        if (classification == "going down"):
            neg+=1
        elif (classification == "going up"):
            pos+=1
        else:
            neu+=1
        i+=1


    maxi = max(neg,pos,neu)

    confidence = maxi / (neg+pos+neu)
    if maxi==neg:
        print("Bearish")
    elif maxi==pos:
        print("Bullish")
    else:
        print("Neutral")

    print("Based on recent news data and sentiment analysis via LLM, we predict that the stock prices of Aptos will behave in the above mentioned way")
    print("The condfidence score is computed based on multiple news instances to provide a generic response as follows- ")
    print("Confidence score:",confidence*100)
    return jsonify(data)

    

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000, debug=True)




text_1 = (
    "Crypto credit card adoption is on the rise in Brazil, issuers have claimed, with more Latin American users than ever now buying and spending tokens."
)

print(classify_review(
    text_1, gpt, tokenizer, device, max_length=1000
))

