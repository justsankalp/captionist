import warnings
warnings.filterwarnings("ignore")

###
import requests
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import pickle
#################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn import TransformerDecoderLayer, TransformerDecoder
#############!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_up_causal_mask(seq_len, device):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
    mask.requires_grad = False
    return mask

class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        """Initializes the module."""
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        """Performs forward pass of the module."""
        skip_connection = x
        x = self.block(x)
        x = skip_connection + x
        return x


class Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(Normalize, self).__init__()
        self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, dim=-1):
        norm = x.norm(2, dim=dim).unsqueeze(-1)
        x = self.eps * (x / norm)
        return x


class PositionalEncodings(nn.Module):
    """Attention is All You Need positional encoding layer"""

    def __init__(self, seq_len, d_model, p_dropout):
        """Initializes the layer."""
        super(PositionalEncodings, self).__init__()
        token_positions = torch.arange(start=0, end=seq_len).view(-1, 1)
        dim_positions = torch.arange(start=0, end=d_model).view(1, -1)
        angles = token_positions / (10000 ** ((2 * dim_positions) / d_model))

        encodings = torch.zeros(1, seq_len, d_model)
        encodings[0, :, ::2] = torch.cos(angles[:, ::2])
        encodings[0, :, 1::2] = torch.sin(angles[:, 1::2])
        encodings.requires_grad = False
        self.register_buffer("positional_encodings", encodings)

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        """Performs forward pass of the module."""
        x = x + self.positional_encodings
        x = self.dropout(x)
        return x


class CaptionDecoder(nn.Module):
    """Decoder for image captions.

    Generates prediction for next caption word given the prviously
    generated word and image features extracted from CNN.
    """

    def __init__(self, padding_idx = 0):
        """Initializes the model."""
        super(CaptionDecoder, self).__init__()
        decoder_layers = 6
        attention_heads = 16
        d_model = 512
        ff_dim = 1024
        dropout = 0.3
        embedding_dim = 300
        vocab_size = 18853
        img_feature_channels = 2048
        max_len = 27


        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.entry_mapping_words = nn.Linear(embedding_dim, d_model)
        self.entry_mapping_img = nn.Linear(img_feature_channels, d_model)

        self.res_block = ResidualBlock(d_model)

        self.positional_encodings = PositionalEncodings(max_len, d_model, dropout)
        transformer_decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(transformer_decoder_layer, decoder_layers)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, image_features, tgt_padding_mask=None, tgt_mask=None):
        """Performs forward pass of the module."""
        # Adapt the dimensionality of the features for image patches
        image_features = self.entry_mapping_img(image_features)
        image_features = image_features.permute(1, 0, 2)
        image_features = F.leaky_relu(image_features)

        # Entry mapping for word tokens
        x = self.embedding_layer(x)
        x = self.entry_mapping_words(x)
        x = F.leaky_relu(x)

        x = self.res_block(x)
        x = F.leaky_relu(x)

        x = self.positional_encodings(x)

        # Get output from the decoder
        x = x.permute(1, 0, 2)
        x = self.decoder(
            tgt=x,
            memory=image_features,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask
        )
        x = x.permute(1, 0, 2)

        x = self.classifier(x)
        return x

#############


class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=400, d_model=512, decoder_layers=6, attention_heads=16, ff_dim=1024, dropout=0.3, padding_idx=0):
        super(CaptionGenerator, self).__init__()

        # Encoder: pretrained ResNet-50
        self.resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the last one
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last layer with a linear layer to output features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-2]))
        max_token_length = 27
        # Decoder
        self.decoder = CaptionDecoder(padding_idx=padding_idx)
        self.causal_mask = set_up_causal_mask(max_token_length, 'cuda')
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, images, x_words, y, tgt_padding_mask=None):

        # Extract image features
        img_features = self.resnet(images)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        image_features = img_features.detach()
        # Pass captions and image features through the decoder
        # predictions = self.decoder(captions[:, :-1], image_features, tgt_padding_mask=tgt_padding_mask, tgt_mask=tgt_mask)

        predictions = self.decoder(x_words, image_features, tgt_padding_mask, self.causal_mask)
        # Calculate the loss
        loss = self.criterion(predictions.view(-1, predictions.size(-1)), y.view(-1))

        return loss, predictions

    def generate_caption(self, image, max_length=20, start_token=1, end_token=2):
  
        # Extract image features
        img_features = self.resnet(image)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        # image_features = img_features.detach()
        # Initialize caption with start token
        caption = torch.tensor([start_token]).long()

        # Generate caption word by word
        for _ in range(max_length):
            # Predict the next word
            prediction = self.decoder(caption.unsqueeze(0), img_features, tgt_mask=None)
            predicted_token = torch.argmax(prediction[0, -1, :]).item()

            # Append predicted token to the caption
            caption = torch.cat([caption, torch.tensor([predicted_token]).long()])

            # Stop if end token is predicted
            if predicted_token == end_token:
                break

        # Convert token indices to words
        caption = [self.vocab.idx2word[token] for token in caption.tolist()]

        # Remove start and end tokens
        caption = caption[1:-1]

        return ' '.join(caption)
################

class vocab():
  def __init__(self, path = "./data/word2idx_space_split.data"):
    self.word2idx = pickle.load(open(path, "rb"))
    self.idx2word = {str(idx): word for word, idx in self.word2idx.items()}

################

def generate_caption(model, image, vocab, max_length=27, start_token=1, end_token=2, pad_token=0):
    model.eval()  # Set the model to evaluation mode

    device = image.device
    with torch.no_grad():
        # Extract image features
        img_features = model.resnet(image.unsqueeze(0))
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)

        # Initialize caption with start token and max_length
        caption = torch.full((max_length,), pad_token, dtype=torch.long).to(device)
        caption[0] = start_token

        padd_mask = torch.full((max_length,), True, dtype=torch.bool).to(device)
        causal_mask = set_up_causal_mask(max_length, device)
        cap_idx = []
        for i in range(max_length - 1):
            padd_mask[i] = False

            # Get the model prediction for the next word
            y_pred_prob = model.decoder(caption.unsqueeze(0), img_features, padd_mask.unsqueeze(0), causal_mask)
            # print(torch.argmax(y_pred_prob[0,i]).item())
            # print(y_pred_prob[0].shape)
            # y_pred_prob = F.softmax(y_pred_prob, dim=-1
            y_pred = torch.argmax(y_pred_prob[0,i]).item()

            # Add the generated word to the caption
            caption[i+1] = y_pred
            cap_idx.append(y_pred)

            if y_pred == end_token:
                break
                print("c= ",caption)

        # Convert token indices to words
        generated_caption = [vocab.idx2word[str(token)] for token in cap_idx]


        return ' '.join(generated_caption[:-2])
##############

vocab_size = 18853
voc = vocab()
model = CaptionGenerator(vocab_size).to(device)
state_dict = torch.load("./models/model-50epochs.pth")
model.load_state_dict(state_dict)

##############
def get_image_from_url(url):

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for unsuccessful requests

        if response.headers.get('Content-Type', '').startswith('image/'):
            # Check for valid image content type
            image_data = BytesIO(response.content)
            img = Image.open(image_data)
            return img
        else:
            raise ValueError(f"Invalid content type: {response.headers['Content-Type']}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

########

from translate import Translator
def translate_text(text, src_lang='kn', dest_lang='en'):
    translator = Translator(from_lang=src_lang, to_lang=dest_lang)
    translated_text = translator.translate(text)
    return translated_text

########
def plot(image, trans):
    plt.imshow(image)
    caps = generate_caption(model, trans.to(device), voc)
    # plt.title(caps)
    print("-------------------------------------------")
    print("Captions in Kannada: ",caps)
    print("Captions in English: ",translate_text(caps))
    plt.show()
    return 0

if __name__ == '__main__':
    while True:
            print("#################################################################")
            option = input("Do you want to load the image \n (0) locally or \n (1) using a URL [0/1]: ")
            if option == '0':
                path = input("Enter the path to Image: ")
                image = Image.open(path).convert('RGB')
            elif option == '1':
                url = input("Enter the URL: ")
                image = get_image_from_url(url).convert('RGB')
    # match option:
    #     case 1:
    #         url = input("Enter the URL: ")
    #         image = get_image_from_url(url).convert('RGB')
    #     case 0:
    #         path = input("Enter the path to Image: ")
    #         image = Image.open(path).convert('RGB')

            preprocessing = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            transformed_image = preprocessing(image)
            plot(image, transformed_image)
            print("#################################################################")


