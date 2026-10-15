"""Denoising backbones — topology-agnostic motion transformers.

One class per point on the two orthogonal axes the config exposes,
``model.attention`` x ``model.text_cond``; the module names spell out the
pair, and :mod:`unimate.models.factory` is what maps a config onto them:

===================  ==========================  ==================
attention/text_cond  module                      class
===================  ==========================  ==================
``full``/``adaln``   :mod:`.full_adaln`          UniMateFullAdaLN
``full``/``cross``   :mod:`.full_cross_attn`     UniMateFullCrossAttn
``graph``/``adaln``  :mod:`.graph_adaln`         UniMateGraphAdaLN
``graph``/``cross``  :mod:`.graph_cross_attn`    UniMateGraphCrossAttn
===================  ==========================  ==================

All four subclass :class:`~.base.UniMateDenoiserBase`, which owns everything
the axes do not touch: config, conditioning embedders, adaLN-Zero init and
classifier-free-guidance masking. The attention blocks they assemble live in
:mod:`.blocks`, positional encodings in :mod:`.rope`.
"""

from .base import UniMateDenoiserBase
from .graph_adaln import UniMateGraphAdaLN

# Trivexa: only the variant of the recommended model (unimate_uniml3d_f60_v3: graph attention, AdaLN text) is kept.
__all__ = [
    "UniMateDenoiserBase",
    "UniMateGraphAdaLN",
]
