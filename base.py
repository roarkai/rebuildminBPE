"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""

import unicodedata

## helper funcitons for Tokenizer.save method
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer:
    """base class for tokenizers"""

    def __init__(self):
        self.merges = {}             # {(int, int): int}
        # self.vocab = self.build_vocab_()
        self.vocab = {}
        self.re_pattern = ""
        self.special_tokens = {}     # {str: int}, e.g. {'<|endoftext|>': 100257}

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError
    
    def _build_vocab(self):
        ## construct self.vocab
        #  self.vocab is in the form of {index: raw bytes} it records 
        #  the information about merged pairs, but unlike the self.merges:
        #  - self.merges: key : value <> (merged_idx1, merged_idx2) : new_idx
        #  - self.vocab:  key : value <> idx : bytes_of_all_ids_merged_in_idx
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
          - contents included:
            - version
            - re_pattern
            - special tokens
            - keys in merges
        - vocab file is just a pretty printed version for human inspection only
          - elemetns in vocab has two type:
            - type_1: {(int_c0, int_c1), int_p}, value of int_p is in 255~vocab_size
            - type_2: {int}, value of int is in the range of 0 ~ 255
          - contents included:
            - type_1: [str_c0, str_c1] -> [str_p] int_p
            - type_2: [str] int
        """
        self._save_model(file_prefix)
        self._save_vocab(file_prefix)

    ## save file_prefix.model
    def _save_model(self, file_prefix):        
        model_file = file_prefix + '.model'

        # given that the contents saved here are all simple characters, 
        # it's not necessay to specify the 'encoding' argument in open()
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write('rk_minbpe v1\n')                # version
            f.write(f'{self.re_pattern}\n')          # re_pattern

            f.write(f'{len(self.special_tokens)}\n') # special_tokens
            for t, idx in self.special_tokens.items():
                f.write(f"{t} {idx}\n")

            for idx1, idx2 in self.merges:           # all keys in merges
                f.write(f'{idx1} {idx2}\n')

    ## save file_prefix.vocab
    def _save_vocab(self, file_prefix):
        vocab_file = file_prefix + '.vocab'
    
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        # given that the str corresponding to bytes in the vocab would include 
        # all kinds of unicode point, open() has to specify the 'encoding' argument
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx, token in self.vocab.items():
                str_idx = render_token(token)
            
            if idx in inverted_merges:
                idx_c1, idx_c2 = inverted_merges[idx]
                str_c1 = render_token(self.vocab[idx_c1])
                str_c2 = render_token(self.vocab[idx_c2])
                f.write(f'[{str_c1}, {str_c2}] -> [{str_idx}] {idx}\n')
            else:
                f.write(f'[{str_idx}] {idx}\n')

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith('model')
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding='utf-8') as f:
            version = f.readline().strip()
            self.re_pattern = f.readline.strip()
            num_special = int(f.readline.strip())
            # read the special tokens
            for _ in range(num_special):
                special, special_idx = f.readline.strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] =idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

    # help function used in self.encode()
    def encode_single_chunk(self, text_in):
        """
        mission: convert a chunk of str string to a list of integers called tokens.
                 the value of each integer(token) is in the range of 0~255.
                 Then the tokens would be processed by a language model.

        inputs:
        @text_in(str string): str string type data. every character is first 
                              converted to a byte string, which then would be split
                              to an integer list with each integer correcpondes to
                              one byte data in the raw byte string.

        output:
        @tok_in(list of intergers): The value of each element is in the range of 
                                    0~vocab_size. And each element is one token of
                                    input that will be processed by a language model.
        """
        # convert the input text from str string to raw bytes
        bytes_in = text_in.encode('utf-8')
        
        # split it to a list with each element stands for a byte in the raw bytes
        idx_in = self.bytes_to_ids(bytes_in)
                
        # execute the merging
        tok_in = merge_all(idx_in, self.merges)

        return tok_in
    
    # GPT4 will override this method to handle its shuffled vocab 
    def bytes_to_ids(self, bytes_in):
        return list(map(int, bytes_in))
    
## helper function used in BasicTokenizer and RegexTokenizer

# used in self.train() and self.merge_all_()
def merge_pair(ids, pair, idx):
    """
    inputs:
    @ids(list of integers): stands for the token state of the training text before 
                            merge. Each pair of consecutive elements will be matched
                            to the target pair @pair. Any matched pair will be
                            replaced by a new index specifiled by @idx.
    @pair(tuple of two integer): target pair of indexes. any pair of consecutive
                                 elements in @ids that has the same value of it
                                 will be replaced.
    @idx(int): a new index that will substitute the @pair in the @ids

    output:
    @new_ids(list of integers): the new token state of the text after every pair of
                                consecutive elements that has the same value as the 
                                @pair has been merged.
    """
    new_ids = []
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    # deal with the could be left single token, the last element in @ids
    if i < len(ids):
        new_ids.append(ids[i])
        i += 1
    return new_ids

# help function used in self.encode_single_chunk()
def merge_all(ids, merges):
    """
    merge all consecutive pair if intergers in @ids that are in the self.merges.

    inputs:
    @ids(list of integers): the value of ach element is in the range of 0~255
    @mergers(dictionary of merged pairs)
        
    output:
    @toks(list of integers): list of tokens that would become the inputs of the LM.
                             The value of each token is in the range of 0~vocab_size. 
    """

    ## a smarter way:
    while len(ids) >= 2:
        #  put all unique consecutive pairs in @ids in a new list
        pairs = get_pairs(ids)
            
        #  find a target pair to be merge, it should be:
        #  1. in the merges, so a corresponding idx is supposed to replace a pair
        #  2. among all pairs, it's the one with the lowest idx in the merges
        tar_pair = min(pairs, key=lambda p: merges.get(p, float('inf')))

        #  test if nothing can be merged
        if tar_pair not in merges:
            break

        #  merge one pair
        tar_idx = merges[tar_pair]
        ids = merge_pair(ids, tar_pair, tar_idx)

    return ids

# help function used in self.train()
def get_stats(ids):
    """
    output:
    @cnt({k=pair: v=frequency_count}): frequency count for each pair of consecutive
                                       elements in @ids
    """
    cnt = {}
    for pair in zip(ids, ids[1:]):
        cnt[pair] = cnt.get(pair, 0) + 1
    return cnt

# help function used in self.encode()
def get_pairs(ids):
    """
    return a list of unique consecutive pairs in @ids
        
    output:
    @pairs: list of index pairs in the @ids
    """
    pairs = []
    for p in zip(ids, ids[1:]):
        pairs.append(p)
    pairs = list(set(pairs))
    return pairs