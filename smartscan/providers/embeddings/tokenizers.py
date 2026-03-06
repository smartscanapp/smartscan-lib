from tokenizers import Tokenizer
from tokenizers.models import WordPiece, BPE
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents

def load_minilm_tokenizer(vocab_path: str):
    tokenizer = Tokenizer(WordPiece.from_file(vocab=vocab_path, unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 101), ("[SEP]", 102)],
    )
    return tokenizer


def load_mpnet_tokenizer(vocab_path: str):
    tokenizer = Tokenizer(WordPiece.from_file(vocab=vocab_path, unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 0), ("[SEP]", 2)],
    )

    return tokenizer

def load_roberta_tokenizer(vocab_path: str, merges_path: str):
    tokenizer = Tokenizer(BPE.from_file(vocab_path, merges_path, unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    return tokenizer

def load_clip_tokenizer(vocab_path: str, merges_path: str):
    tokenizer = Tokenizer(BPE.from_file(vocab_path, merges_path))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<start_of_text> $A <end_of_text>",
        pair="<start_of_text> $A <end_of_text> $B:1 <end_of_text>:1",
        special_tokens=[
            ("<start_of_text>", 49406),
            ("<end_of_text>", 49407),
        ],
    )
    return tokenizer
