# -*- coding: utf-8 -*-
# @Time   : 2024/09/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append("../..")

from transformers import LlamaTokenizerFast

from scarabs.tokenizer_factory import (
    TokenizerFactoryWithBPE,
    TokenizerFactoryWithSentencePiece,
)

data_files = "../data/pretrain/pretrain.jsonl"

# twsp = TokenizerFactoryWithSentencePiece(
#     model_type="bpe",
#     vocab_size=38195,
#     special_tokens=["<s>", "</s>"],
#     output_model_prefix="./tokenizer",
# )

# twsp.create_tokenizer(data_files)

# twsp.load_tokenizer()

# mk = twsp.encode("你好，世界！")
# print(mk)
# print(twsp.decode(mk))


twsp = TokenizerFactoryWithBPE(
    unk_token="<unk>",
    vocab_size=50000,
    special_tokens=["<unk>", "<s>", "</s>"],
    TokenizerFastFunc=LlamaTokenizerFast,
    output_dir="./tokenizer",
)

twsp.create_tokenizer([data_files])

twsp.load_tokenizer()

mk = twsp.encode("你好，世界！")
print(mk)
print(twsp.decode(mk))
