# -*- coding: utf-8 -*-
# @Time   : 2024/09/12 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition

from __future__ import absolute_import, division, print_function

import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import LlamaTokenizerFast


class TokenizerFactory:
    def __init__(self) -> None:
        pass

    def create_tokenizer(self):
        pass


class TokenizerFactoryWithBPE(TokenizerFactory):
    def __init__(
        self, unk_token, vocab_size, special_tokens, TokenizerFastFunc, output_dir
    ):
        self.unk_token = unk_token
        self.TokenizerFastFunc = TokenizerFastFunc
        self.output_dir = output_dir
        #
        self.tokenizer = Tokenizer(
            BPE(unk_token=unk_token, fuse_unk=True, byte_fallback=True)
        )
        self.tokenizer.normalizer = Sequence([NFKC()])  # type:ignore
        self.tokenizer.pre_tokenizer = ByteLevel()  # type:ignore
        self.tokenizer.decoder = ByteLevelDecoder()  # type:ignore

        # special_tokens = ["<unk>", "</s>", "<s>"]  # type:ignore
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,  # type:ignore
            show_progress=True,  # type:ignore
            inital_alphabet=ByteLevel.alphabet(),  # type:ignore
            special_tokens=special_tokens,  # type:ignore
            max_token_length=4,  # type:ignore
        )

    def create_tokenizer(self, data_files):
        self.tokenizer.train(data_files, self.trainer)
        ntk = self.TokenizerFastFunc(tokenizer_object=self.tokenizer, legacy=False)
        ntk.save_pretrained(self.output_dir)

    def load_tokenizer(self):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.output_dir)

    def encode(self, x):
        return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)


class TokenizerFactoryWithSentencePiece(TokenizerFactory):
    def __init__(self, model_type, vocab_size, special_tokens, output_model_prefix):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.output_model_prefix = output_model_prefix

    def create_tokenizer(self, data_files):
        # # 训练 SentencePiece 模型
        spm.SentencePieceTrainer.train(  # type:ignore
            input=data_files,
            model_prefix=self.output_model_prefix,
            vocab_size=self.vocab_size,  # 词汇表大小
            model_type=self.model_type,  # 模型类型（unigram, bpe, char, word）
            shuffle_input_sentence=True,  # 是否打乱输入句子
            normalization_rule_name="nfkc",
            user_defined_symbols=self.special_tokens,  # 自定义符号
        )

    def load_tokenizer(self):
        self.sp = spm.SentencePieceProcessor(
            model_file=f"{self.output_model_prefix}.model"  # type:ignore
        )

    def encode(self, x):
        return self.sp.encode_as_ids(x)  # type:ignore

    def decode(self, x):
        return self.sp.decode_ids(x)  # type:ignore
