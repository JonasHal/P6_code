import pandas as pd
import numpy as np
from pathlib import Path
import math
import torch
import torch.nn as nn
from torch import nn, Tensor
from sklearn.preprocessing import MinMaxScaler

# Creating the inputs for the transformer model
def get_src_trg(
    sequence: torch.Tensor,
    enc_seq_len: int,
    dec_seq_len: int,
    target_seq_len: int
    ):

    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence.
    Args:
        sequence: tensor, a 1D tensor of length n where
                n = encoder input length + target sequence length
        enc_seq_len: int, the desired length of the input to the transformer encoder
        target_seq_len: int, the desired length of the target sequence (the
                        one against which the model output is compared)
    Return:
        src: tensor, 1D, used as input to the transformer model
        trg: tensor, 1D, used as input to the transformer model
        trg_y: tensor, 1D, the target sequence against which the model output
            is compared when computing loss.
    """
    assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

    # encoder input
    src = sequence[:enc_seq_len]

    # decoder input. As per the paper, it must have the same dimension as the
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len-1:len(sequence)-1]

    assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:]

    assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

    return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]

def generate_square_subsequent_mask(dim1: int, dim2: int, dim3: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Args:
        dim1: int, batch_size * n_heads
        dim2: int. For src and trg masking this must be target sequence length.
        dim3: int. For src masking, this must be encoder sequence length.
              For trg masking, this must be target sequence length
    Return:
        A Tensor of shape [dim1, dim2, dim3]
    """
    return torch.triu(torch.ones(dim1, dim2, dim3) * float('-inf'), diagonal=1)


class PositionalEncoder(nn.Module):
    """
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    https://github.com/LiamMaclean216/Pytorch-Transfomer/blob/master/utils.py
    """

    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):
        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values
        # dependent on position and i
        position = torch.arange(max_seq_len).unsqueeze(1)

        exp_input = torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)

        div_term = torch.exp(exp_input)  # Returns a new tensor with the exponential of the elements of exp_input

        pe = torch.zeros(max_seq_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)  # torch.Size([target_seq_len, dim_val])

        pe = pe.unsqueeze(0).transpose(0, 1)  # torch.Size([target_seq_len, input_size, dim_val])

        # register that pe is not a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """

        add = self.pe[:x.size(1), :].squeeze(1)

        x = x + add

        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".
    A detailed description of the code can be found in my article here:
    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.
    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.
    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020).
    'Deep Transformer Models for Time Series Forecasting:
    The Influenza Prevalence Case'.
    arXiv:2001.08317 [cs, stat] [Preprint].
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).
    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).
    """

    def __init__(self,
        input_size: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int=58,
        dim_val: int=512,
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2,
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.2,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        ):

        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            max_seq_len: int, length of the longest sequence the model will
                         receive. Used in positional encoding.
            out_seq_len: int, the length of the model's output (i.e. the target
                         sequence length)
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        print("input_size is: {}".format(input_size))
        print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
            )

        self.decoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
            )

        self.linear_mapping = nn.Linear(
            in_features=out_seq_len*dim_val,
            out_features=out_seq_len
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=max_seq_len
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True
            )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
            )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
                tgt_mask: Tensor=None) -> Tensor:
        """
        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the feature number
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, E is the feature number.
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(
            src=src
            )

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt)

        # Pass throguh decoder
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        # Pass through the linear mapping layer
        decoder_output= self.linear_mapping(decoder_output.flatten(start_dim=1))

        return decoder_output

if __name__ == "__main__":
    data = pd.read_csv(Path('total_data.csv'))
    data = data[["total_kWhDelivered"]]
    mm = MinMaxScaler(feature_range=(0, 1))
    data = torch.tensor(mm.fit_transform(data).astype(np.float32))

    ## Model parameters
    dim_val = 512  # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
    n_heads = 8  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
    n_decoder_layers = 4  # Number of times the decoder layer is stacked in the decoder
    n_encoder_layers = 4  # Number of times the encoder layer is stacked in the encoder
    input_size = 1  # The number of input variables. 1 if univariate forecasting.
    dec_seq_len = 92  # length of input given to decoder. Can have any integer value.
    enc_seq_len = 183  # length of input given to encoder. Can have any integer value.
    output_sequence_length = 48  # Length of the target sequence, i.e. how many time steps should your forecast cover
    max_seq_len = enc_seq_len  # What's the longest sequence the model will encounter? Used to make the positional encoder

    model = TimeSeriesTransformer(
        dim_val=dim_val,
        input_size=input_size,
        dec_seq_len=dec_seq_len,
        max_seq_len=max_seq_len,
        out_seq_len=output_sequence_length,
        n_decoder_layers=n_decoder_layers,
        n_encoder_layers=n_encoder_layers,
        n_heads=n_heads)

    # Define some hyperparameters
    target_seq_len = 48  # the desired length of the model forecast

    # Make a sub-sequence from the data
    sub_sequence = data[0:enc_seq_len+output_sequence_length]

    # Get src, trg, trg_y
    src, trg, trg_y = get_src_trg(
        sequence=data,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=target_seq_len
    )

    #For the masks
    batch_size = 24

    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = generate_square_subsequent_mask(
        dim1=batch_size * n_heads,
        dim2=output_sequence_length,
        dim3=enc_seq_len
    )

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = generate_square_subsequent_mask(
        dim1=batch_size * n_heads,
        dim2=output_sequence_length,
        dim3=output_sequence_length
    )

    output = model(
        src=src,
        tgt=trg,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )
