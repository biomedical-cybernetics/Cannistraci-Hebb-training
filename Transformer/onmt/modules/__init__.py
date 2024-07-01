"""  Attention and normalization modules  """
from onmt.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute

import onmt.modules.source_noise  # noqa
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.average_attn import AverageAttention
from onmt.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding

__all__ = ["CopyGenerator", 
           "CopyGeneratorLoss", "CopyGeneratorLossCompute", "MultiHeadedAttention", "AverageAttention", "Embeddings", "PositionalEncoding"]
