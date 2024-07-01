"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase
from onmt.decoders.transformer import TransformerDecoder, CHTTransformerDecoder


str2dec = {"transformer": TransformerDecoder, "chttransformer": CHTTransformerDecoder}

__all__ = ["DecoderBase",
           "str2dec",
           "TransformerDecoder",
           "CHTTransformerDecoder"]
