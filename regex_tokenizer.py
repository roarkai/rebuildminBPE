from base import Tokenizer, merge_pair
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, re_pat=None):
        super().__init__()
        self.re_pat = re_pat
        self.reverse_special_tokens = {}

    def regst_special_tokens(self, special_tokens):
        """
        @special_tokens: {str: int}, eg: {'<|endoftext|>' : 100257}
        """ 
        self.special_tokens = special_tokens
        self.inverse_special_tokens = { v: k for k, v in special_tokens.items() }

    def train(self, text, vocab_size, verbose=False):

        assert vocab_size >= 256
        # 1. split text according to the regex pattern
        split_text = re_split(text, self.re_pat)

        # 2. convert elements type from str string to indexes
        #    - each element is converted to a list of integers
        #    - @split_ids is a list of list
        split_ids = [list(map(int, i.encode('utf-8'))) for i  in split_text]

        # 3. merge and record the merged pairs in the self.merges dict
        num_merge = 0
        while num_merge < vocab_size - 256:
            # 1) get stats
            cnts = self._get_split_stats(split_ids)
            if not cnts:
                raise ValueError('vocab_size too large! all chunks\'re merged to one token')
            # print(f'cnts in this turn:\n{cnts}')       # for debug

            # 2) get the most frequent pair
            candi_pair = max(cnts, key=cnts.get)

            # 3) execute merge
            new_idx = 256 + num_merge
            split_ids = self._merge_split_pair(split_ids, candi_pair, new_idx)

            # 4) update self.merges
            self.merges[candi_pair] = new_idx

            # 5) update num_merge
            num_merge += 1

            # 6) (optional)print info. for debug
            if verbose:
                print(f'merge {num_merge}/{vocab_size-256}: {candi_pair} => {new_idx}')
                print(split_ids)

        # 4. construct self.vocab
        self.vocab = self._build_vocab()

    def _get_split_stats(self, split_ids):
        cnts = {}
        for i in split_ids:
            if len(i) > 1:
                for pair in zip(i, i[1:]):
                    cnts[pair] = cnts.get(pair, 0) + 1
        return cnts
    
    def _merge_split_pair(self, split_ids, candi_pair, new_idx):
        new_ids = []
        for i in split_ids:
            new_i = merge_pair(i, candi_pair, new_idx)
            new_ids.append(new_i)
        return new_ids

    def encode(self, text_in, allowed_special='none'):
        """
        encode text_in. convert the str string type input corpus to a list of integers called tokens.
        unlike encode() in BasicTokenizer, special tokens needs to be handled here

        inputs:
        @allowed_special: used to specify how to handle special tokens. There are four values for it:
                          1. 'all': all special tokens in the tokenizer will be used
                          2. 'none': no special tokens will be handled, even if there are special
                                     tokens in the text, which means special tokens will be treated 
                                     the same as normal characters.
                          3. 'none_raise': no special tokens will be used, but need to make sure that
                                           there are no special tokens in the text. otherwise a 
                                           ValueError will be raised.
                          4. a set of special tokens: only the specified spceial tokens will be handled
                                                      as special tokens. if there are other special tokens
                                                      in the text that are not in the set but are in the 
                                                      tokenizer model, they will be handled as normal characters.
        @text_in: str string type data composed of normal character and special tokens. The two kinds
                  of tokens are encoded seperately. the procedure is:
                  1. split the text apart with consecutive normal characters in a single chunk and 
                     each special token is an independent chunk.
                  2. encode those two kinds of chunks using different methods.
                     - For each normal character chunks:
                       1) Every character is converted to (utf-8) raw bytes
                       2) The whole raw bytes are split into an list of integers, with each int 
                       corresponding to one byte of (utf-8) raw bytes.
                       3) merge the consecutive pairs of elements in the list if they are in the key
                       of @self.merges
                     - For each speical tokens:
                       Every special token is directyly converted to its number in the self.vocab
        """
        ## handle the configuration of special tokens
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(t not in text_in for t in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f'allowed_special={allowed_special} not legal')
        
        ## if no special tokens need to be handled, encode the text directly.
        if not special:
            return self.encode_no_special(text_in)
        
        ## if special tokens need to be handled, then:
        #  1. split consecutive normal characters and special tokens into chunks
        special_re_pattern = '(' + '|'.join(re.escape(k) for k in special) + ')'
        chunks = re.split(special_re_pattern, text_in)

        #  2. handle each chunk in the splitted texts according to its type
        ids = []
        for c in chunks:
            if c in special:  # convert special token to its integer number
                ids.append(special[c])
            else:             # convert normal character chunk
                ids.extend(self.encode_no_special(c))
        return ids


    def encode_no_special(self, text_in):
        """
        Encode the text composed only of normal characters, with no special tokens.
        The text string is split into chunks according to the regex pattern. 
        Then, each chunk is converted into a list of integers and merged.
        Finally, all merged lists are combined into a single list.

        inputs:
        @text_in(str string): str string type data. every character is first converted to a 
                              byte string, which then would be split to an integer list with 
                              each integer correcpondes to one byte data in the raw byte string.

        output:
        @tok_in(list of intergers): The value of each element is in the range of 0~vocab_size.
                                    And each element is one token of input that would be 
                                    processed by a language model.
        """
        # 1. split according to the regex pattern
        reged_txt = re_split(text_in, self.re_pat)

        # 2. convert, merge and combine
        enc_txt = []
        for t in reged_txt:
            enc_t = self.encode_single_chunk(t)
            enc_txt.extend(enc_t)

        return enc_txt


    def decode(self, toks_out):
        """
        mission: convert the outputs of LM @toks_out from token list to str string.
                 besides ops in BasicTokenizer.decode, ops for special tokens is needed here.
        input:
        @toks_out(list of integers): It is the output token list of LM. 
                                     int in the list is in the range of 0~vocab_size
        """
        bytes_out = b''
        for idx in toks_out:
            if idx in self.vocab:
                bytes_out += self.vocab[idx]
            elif idx in self.inverse_special_tokens:
                bytes_out += self.inverse_special_tokens[idx].encode('utf-8')
            else:
                raise ValueError(f'invalid token id: {idx}')
        txt_out = bytes_out.decode(encoding='utf-8', errors='replace')
        return txt_out

# help function used in self.encode()
def re_split(text, re_pat):
    """
    using the regex pattern to split the text into a list of short strings

    input:
    @text(str string): original text

    output:
    @reged_txt(list of str strings)
    """
    pat = re.compile(re_pat)
    reged_txt = re.findall(pat, text)
    # print(reged_txt)   # for debug
    return reged_txt