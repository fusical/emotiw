import torch
import pickle
import pandas as pd

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
import numpy as np
import warnings
from tqdm import tqdm_notebook as tqdm
from typing import List, Tuple

NUM_MAX_POSITIONS = 256
BATCH_SIZE = 32

class TextProcessor:
    
    # special tokens for classification and padding
    CLS = '[CLS]'
    PAD = '[PAD]'
    
    def __init__(self, tokenizer, num_max_positions:int=512):
        self.tokenizer=tokenizer
        self.num_max_positions = num_max_positions
        
    
    def process_example(self, example: Tuple[str, str]):
        "Convert text (example[0]) to sequence of IDs and label (example[1] to integer"
        assert len(example) == 2
        label, text = example[0], example[1]
        assert isinstance(text, str)
        tokens = self.tokenizer.tokenize(text)

        # truncate if too long
        if len(tokens) >= self.num_max_positions:
            tokens = tokens[:self.num_max_positions-1] 
            ids =  self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.CLS]]
        # pad if too short
        else:
            pad = [self.tokenizer.vocab[self.PAD]] * (self.num_max_positions-len(tokens)-1)
            ids =  self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.CLS]] + pad
        
        return np.array(ids, dtype='int64'), 0
    
# download the 'bert-base-cased' tokenizer
from pytorch_transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

# initialize a TextProcessor
processor = TextProcessor(tokenizer, num_max_positions=NUM_MAX_POSITIONS)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

num_cores = cpu_count()

def process_row(processor, row):
    return processor.process_example((row[1][LABEL_COL], row[1][TEXT_COL]))

def create_dataloader(df: pd.DataFrame,
                      processor: TextProcessor,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      text_col: str = "text",
                      label_col: str = "label"):
    "Process rows in `df` with `processor` and return a  DataLoader"

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        result = list(
            tqdm(executor.map(process_row,
                              repeat(processor),
                              df.iterrows(),
                              chunksize=len(df) // 10),
                 desc=f"Processing {len(df)} examples on {num_cores} cores",
                 total=len(df)))

    features = [r[0] for r in result]
    labels = [r[1] for r in result]

    dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                            torch.tensor(labels, dtype=torch.long))

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

import torch.nn as nn

def get_num_params(model):
    mp = filter(lambda p: p.requires_grad, model.parameters())
    return sum(np.prod(p.size()) for p in mp)

class Transformer(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"

    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithClfHead(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"
    def __init__(self, config, fine_tuning_config):
        super().__init__()
        self.config = fine_tuning_config
        self.transformer = Transformer(config.embed_dim, config.hidden_dim, config.num_embeddings,
                                       config.num_max_positions, config.num_heads, config.num_layers,
                                       fine_tuning_config.dropout, causal=not config.mlm)
        
        self.classification_head = nn.Linear(config.embed_dim, fine_tuning_config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, clf_tokens_mask, clf_labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)

        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(clf_logits.view(-1, clf_logits.size(-1)), clf_labels.view(-1))
            return clf_logits, loss
        return clf_logits


class ImageCaptioningClassifier:
    """
    Classifies sentiment based on image captions extracted from frames extracted from video clips
    NOTE: This classifier uses PyTorch, unlike most classifiers in this package. 
    """

    def __init__(self, caption_pkl, caption_pkl_key_prefix="train_", model_metadata_location=None, model_location=None, is_test=None, frames_to_use=12, batch_size=16):
        self.caption_pkl = caption_pkl
        self.caption_pkl_key_prefix = caption_pkl_key_prefix
        self.is_test = is_test
        self.model_metadata_location = model_metadata_location
        self.model_location = model_location
        self.frames_to_use = frames_to_use
        self.batch_size = batch_size
        print(f"ImageCaptioningClassifier created with caption_pkl = {caption_pkl} , is_test = {is_test} , model_location = {model_location}")

        if "https://" in self.model_location or "http://" in self.model_location:
            self.metadata = torch.hub.load_state_dict_from_url(self.model_metadata_location)
            self.model = TransformerWithClfHead(self.metadata["config"], self.metadata["config_ft"]).to(self.metadata["config_ft"].device)
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(self.model_location))
        else:
            self.metadata = torch.load(self.model_metadata_location)
            self.model = TransformerWithClfHead(self.metadata["config"], self.metadata["config_ft"]).to(self.metadata["config_ft"].device)
            self.model.load_state_dict(torch.load(self.model_location))

    def predict(self, layer=None):
        """
        Performs sentiment classification prediction on preprocessed frames files
        @param layer: If None, performs normal sentiment classification.
                      If not None, returns the values from the intermediate layers.
        return:
            - The model prediction result
            - The video file names for each of the rows returned in model.predict
              (without the .mp4 suffix)
        """
        with open(self.caption_pkl, "rb") as f:
            train_captions_obj = pickle.load(f)

        train_captions = []
        for tc in train_captions_obj[caption_pkl_key_prefix + "captions"]:
            # For each video
            sentences = []
            for t in tc:
                # For each frame, find the best caption (caption with the max log prob)
                best_caption = ""
                best_log_prob = -100

                for candidate in t:
                    if candidate["log_prob"] > best_log_prob:
                        best_log_prob = candidate["log_prob"]
                        best_caption = candidate["caption"]

                best_caption = best_caption.replace(" .", "")
                sentences.append(best_caption)
                break
            train_captions.append("".join(sentences))

        df_train = pd.DataFrame(list(zip(train_captions, train_captions_obj[caption_pkl_key_prefix + "videos"])), columns =['text', 'vid_name']) 
        train_dl = create_dataloader(df_train, processor, 
                                    batch_size=32, 
                                    shuffle=False)

        if layer is None:
            ### Make sure evaluation works on restored model
            pred_labels = []

            for batch in train_dl:
                self.model.eval()
                with torch.no_grad():
                    batch, labels = (t.to(self.metadata["config_ft"].device) for t in batch)
                    inputs = batch.transpose(0, 1).contiguous()
                    logits = self.model(inputs,
                                    clf_tokens_mask = (inputs == tokenizer.vocab[processor.CLS]),
                                    padding_mask = (batch == tokenizer.vocab[processor.PAD]))
                    pred_labels.extend(logits.argmax(axis=1).tolist())
            return np.array(pred_labels), df_train["vid_name"].tolist()
        else:
            pred_labels = torch.zeros((len(df_train), 410))
            i = 0
            for batch in train_dl:
                self.model.eval()
                with torch.no_grad():
                    batch, labels = (t.to(finetuning_config.device) for t in batch)
                    inputs = batch.transpose(0, 1).contiguous()
                    logits = self.model(inputs,
                                    clf_tokens_mask = (inputs == tokenizer.vocab[processor.CLS]),
                                    padding_mask = (batch == tokenizer.vocab[processor.PAD]))
                    begin_index = i * 32
                    end_index = begin_index + len(labels)
                    pred_labels[begin_index:end_index, :] = logits
                i += 1
            return pred_labels.numpy(), df_train["vid_name"].tolist()

    def summary(self):
        """
        Summarizes the pre-trained model
        """
        print(self.model)

    def evaluate(self):
        """
        Evaluates the frames files on the pre-trained model
        return: The evaluation results
        """
        print("Evaluation not supported yet")
        return
