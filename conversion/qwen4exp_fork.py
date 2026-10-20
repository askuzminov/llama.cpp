from __future__ import annotations

# The fork's qwen4exp converter: the upstream class (conversion/qwen4exp.py), which exports the MTP draft head itself
# since c061df198 and writes the same metadata, plus fallbacks for MTP names its filter does not catch (a `model.mtp.`
# prefix). conversion/__init__.py imports this module after the upstream one, so the registration below replaces the
# upstream text model. scripts/fork/copies.txt tracks the upstream file it follows.

from typing import Callable

import torch
from torch import Tensor

from .base import ModelBase
from .qwen4exp import Qwen4ExpTextModel


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
class Qwen4ExpTextModelFork(Qwen4ExpTextModel):
    """The checkpoint also carries a NextN/MTP draft head under `mtp.*`, exported as a
    trailing block; pass --no-nextn to leave it out.
    """

    supports_mtp_export = True
    no_mtp = False

    # The MTP head is one trunk-shaped block (full attention + MoE, wrapped in
    # hyper-connections) plus a combiner, so once _QwenMtpMixin renames
    # `mtp.layers.0.*` to the trailing block index its tensors ride the existing
    # qwen4exp mappings unchanged. Only the two head-level pieces below differ.

    _MTP_MIXER_PREFIX = "mtp.hyper_connection_mixer."

    @classmethod
    def filter_tensors(cls, item):
        # the head carries its own copy of the trunk's hc_head_* output mixer,
        # which qwen4exp has in place of a final norm; it is unindexed in the
        # checkpoint and per-block in the GGUF
        name, gen = item
        if name.startswith("model." + cls._MTP_MIXER_PREFIX):
            name = name.replace("model.", "", 1)
        if name.startswith(cls._MTP_MIXER_PREFIX):
            if cls.no_mtp:
                return None
            assert cls._original_block_count is not None
            return f"model.layers.{cls._original_block_count}.{name[len('mtp.'):]}", gen
        return super().filter_tensors((name, gen))

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        # qwen4exp splits the combiner the shared NextN code calls eh_proj into
        # fc_embedding and fc_hidden; W_e@e + W_h@h == [W_e|W_h] @ concat(e, h),
        # so the two fuse back into the single expected matmul
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)

        emb = tensors.pop("mtp.fc_embedding.weight", None)
        hid = tensors.pop("mtp.fc_hidden.weight", None)
        if emb is None and hid is None:
            return tensors
        if emb is None or hid is None:
            raise ValueError(
                "the qwen4exp MTP combiner needs both mtp.fc_embedding.weight and "
                "mtp.fc_hidden.weight; pass --no-nextn to convert without the draft head"
            )

        assert self._original_block_count is not None
        # fc_embedding first: the graph concatenates the token embedding ahead of
        # the hidden state, so the fused weight has to be ordered to match
        name = f"model.layers.{self._original_block_count}.eh_proj.weight"
        tensors[name] = lambda: torch.cat([emb(), hid()], dim=1)
        return tensors
