"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder, CHTTransformerEncoder


str2enc = {"transformer": TransformerEncoder, "chttransformer": CHTTransformerEncoder}

__all__ = ["EncoderBase", "str2enc",
           "CHTTransformerDecoder"]
