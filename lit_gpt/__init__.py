# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

from importlib import import_module

__all__ = ["GPT", "Config", "Tokenizer", "FusedCrossEntropyLoss"]


def __getattr__(name):
    if name == "GPT":
        return import_module("lit_gpt.model").GPT
    if name == "Config":
        return import_module("lit_gpt.config").Config
    if name == "Tokenizer":
        return import_module("lit_gpt.tokenizer").Tokenizer
    if name == "FusedCrossEntropyLoss":
        return import_module("lit_gpt.fused_cross_entropy").FusedCrossEntropyLoss
    raise AttributeError(f"module 'lit_gpt' has no attribute {name!r}")
