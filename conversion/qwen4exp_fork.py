from __future__ import annotations

# The fork's qwen4exp converter: the upstream class (conversion/qwen4exp.py) plus the MTP draft head, exported as a
# trailing block. conversion/__init__.py imports this module after the upstream one, so the registration below
# replaces the upstream text model. scripts/fork/copies.txt tracks the upstream file it follows.

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

    # The MTP head is one trunk-shaped block (dense attention + MoE, wrapped in
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

    # the upstream method with the MTP block in the ratios and no PLE keys for a draft-only export. It skips the
    # upstream body, which writes the same keys except the ratio of the MTP block (upstream: the trunk ratio, a QSA
    # block; here 0, dense). Since upstream c061df198 the upstream filter_tensors renames mtp.fc_* and fuses them in
    # modify_tensors, so the fusion in index_tensors above only runs for names that filter does not match
    def set_gguf_parameters(self):
        super(Qwen4ExpTextModel, self).set_gguf_parameters()
        hp = self.hparams

        self.gguf_writer.add_hyper_connection_count(hp["hc_count"])
        self.gguf_writer.add_hyper_connection_low_rank(hp["hc_lowrank"])

        n_layer = hp["num_hidden_layers"]
        self.gguf_writer.add_indexer_head_count(hp["indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hp["indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hp["indexer_budget"])
        ratio = hp["indexer_compress_ratio"]
        layer_types = hp["layer_types"]
        ratios = [ratio if layer_types[i] == "full_attention" else 0 for i in range(n_layer)]
        # llama.cpp reads this array with length block_count, and the MTP block trailing the
        # trunk attends densely, which is what a ratio of 0 selects
        ratios += [0] * (self.block_count - n_layer)
        self.gguf_writer.add_attention_compress_ratios(ratios)

        # ple_layer_ids is 1-based in the HF config; empty means no n-gram table,
        # so emit no PLE keys rather than optional ones. A draft-only export carries no
        # trunk tensors, so it carries no PLE table to describe either.
        ple_layers = [i - 1 for i in hp["ple_layer_ids"]]
        if not ple_layers or self.mtp_only:
            return
        self.gguf_writer.add_ple_layers(ple_layers)
        self.gguf_writer.add_ple_ngram_size(hp["ngram_size"])
        self.gguf_writer.add_ple_heads_per_ngram(hp["heads_per_ngram"])
        self.gguf_writer.add_ple_conv_kernel(hp["ple_conv_kernel_size"])
        self.gguf_writer.add_ple_eos_token_id(self._eos_token_id())
        # an image is decoded as an embeddings-only batch, so the graph has no placeholder
        # ids to hash; carry the id and let it stand in for those positions
        _img = self._image_token_id()
        if _img is not None:
            self.gguf_writer.add_ple_image_token_id(int(_img))
        if self._ple_row_dim is not None:
            self.gguf_writer.add_embedding_length_per_layer_input(self._ple_row_dim)

        self.gguf_writer.add_ple_layer_multipliers(
            self._read_hash_constants("ple_embedding.layer_multipliers"))
        self.gguf_writer.add_ple_head_offsets(
            self._read_hash_constants("ple_embedding.ngram_heads_offsets"))
        self.gguf_writer.add_ple_head_vocab_sizes(
            self._read_hash_constants("ple_embedding.ngram_heads_vocab_sizes"))
