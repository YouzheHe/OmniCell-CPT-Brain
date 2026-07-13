"""Hugging Face OmniCell package."""

from .api import OmniCell, OmniCellPipeline, embed_h5ad
from .configuration_omnicell import OmniCellConfig
from .data import GeneVocab, OmniCellDataCollator, OmniCellH5ADDataset, OmniCellH5ADTokenizer
from .legacy import build_model_from_legacy, load_legacy_config, load_legacy_state_dict
from .modeling_omnicell import (
    OmniCellForSupervisedFineTuning,
    OmniCellForUnsupervisedFineTuning,
    OmniCellModel,
)
from .processing_omnicell import OmniCellProcessor

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register(OmniCellConfig.model_type, OmniCellConfig)
    AutoModel.register(OmniCellConfig, OmniCellModel)
except Exception:
    pass

__all__ = [
    "GeneVocab",
    "OmniCell",
    "OmniCellConfig",
    "OmniCellDataCollator",
    "OmniCellForSupervisedFineTuning",
    "OmniCellForUnsupervisedFineTuning",
    "OmniCellH5ADDataset",
    "OmniCellH5ADTokenizer",
    "OmniCellModel",
    "OmniCellPipeline",
    "OmniCellProcessor",
    "build_model_from_legacy",
    "embed_h5ad",
    "load_legacy_config",
    "load_legacy_state_dict",
]
