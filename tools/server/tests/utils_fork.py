#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# fork: ServerProcess options for the context checkpoints and the prompt cache spill, see utils.py


class ServerProcessFork:
    n_ctx_checkpoints: int | None = None
    checkpoint_min_step: int | None = None
    cache_ram_reserve: int | None = None
    cache_spill_dir: str | None = None
    cache_disk: int | None = None
    cache_min_tokens: int | None = None
    cache_idle_write: bool | None = None
    pool_file: str | None = None

    def fork_server_args(self) -> list:
        server_args = []
        if self.n_ctx_checkpoints is not None:
            server_args.extend(["--ctx-checkpoints", self.n_ctx_checkpoints])
        if self.checkpoint_min_step is not None:
            server_args.extend(["--checkpoint-min-step", self.checkpoint_min_step])
        if self.cache_ram_reserve is not None:
            server_args.extend(["--cache-ram-reserve", self.cache_ram_reserve])
        if self.cache_spill_dir is not None:
            server_args.extend(["--cache-spill-dir", self.cache_spill_dir])
        if self.cache_disk is not None:
            server_args.extend(["--cache-disk", self.cache_disk])
        if self.cache_min_tokens is not None:
            server_args.extend(["--cache-min-tokens", self.cache_min_tokens])
        if self.cache_idle_write:
            server_args.append("--cache-idle-write")
        if self.pool_file is not None:
            server_args.extend(["--pool", self.pool_file])
        return server_args
