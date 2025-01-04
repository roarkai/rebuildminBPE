"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""

import tiktoken
from regex_tokenizer import RegexTokenizer

# helper function used in recover_merges()
def bpe(mergable_ranks, token, max_rank):
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = min_rank = None

        # find the pair that has the lowest rank
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        
        # if len(parts) == 2, stop loop
        if min_rank == max_rank:
            break

        # otherwise, continue the merge loop
        parts = parts[:min_idx] + \
                [parts[min_idx] + parts[min_idx + 1]] + \
                parts[min_idx + 2:]
    
    return parts

# recover the dictionary of merges
def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

def render_token(t: bytes) -> str:

    pass

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """
    lightweight wrapper on RegexTokenizer that matchs the GPT4's tokenizer.
    """
    def __init__(self, re_pat=GPT4_SPLIT_PATTERN):
        super().__init__(re_pat)

        # 1. get merge info. from tiktoken
        enc = tiktoken.get_encoding('cl100k_base')
        mergeable_ranks = enc._mergeable_ranks
        
        # 2. recover the merges of GPT4
        self.merges = recover_merges(mergeable_ranks)

        # 3. recover the vocab of GPT4
        #    the index of all individual bytes in the vocab of GPT series are shuffled. So to
        #    they are corresponding to the first 256 elements in the vocab.
        #    [ref to] the encoder.py in gpt2 files, and load.py in tiktoken files
        #    [rk's guess] it's for better visualization of bytes activity in the model
        self.vocab = self._build_vocab()
        #    check the recovered vocab file matches the mergeable_ranks
        vocab_direct = {v: k for k, v in mergeable_ranks.items()}
        assert self.vocab == vocab_direct

        # 5. register special tokens
        self.regst_special_tokens(GPT4_SPECIAL_TOKENS)

    def _build_vocab(self):
        return build_shuffled_vocab(self.merges)

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    # helper function used during 'bytes_in -> ids_in' for handling the shuffled vocab
    # override the corresponding method used in encode
    def bytes_to_ids(self, bytes_in):
        ids_in = self._bytes_to_ids_shuffled(bytes_in)
        return ids_in

    def _bytes_to_ids_shuffled(self, bytes_in):
        ids_shuffled = bytes_to_unicode().keys()
        bytes_shuffled_to_ids = {bytes([b]): i for i, b in enumerate(ids_shuffled)}
        ids_in = [bytes_shuffled_to_ids[bytes([b])] for b in bytes_in]
        return ids_in


## helper function used for rebuild vocab
#  this is how gpt series shuffle the first 256 elements in the vocab
#  also a helper function for token visualization
def bytes_to_unicode():
    # bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    bs = [b for b in range(256) if chr(b).isprintable() and chr(b) != ' ']
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# recover the vocab in GPT4
def build_shuffled_vocab(merges):
    vocab = {}
    # shuffle the first 256 elements in the vocab the same way as in GPT4 
    byte_unicode = bytes_to_unicode()
    for i, bi in enumerate(byte_unicode):
        vocab[i] = bytes([bi])
    
    # starting from index 256, use items in merges to recover the whole vocab
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab