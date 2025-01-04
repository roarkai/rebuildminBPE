"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from base import Tokenizer, merge_pair, get_stats

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        """
        inputs:
        - @text(str string) string of input text
        - @vocab_size(int) the lenghth of the vocabulary
        
        outputs:
        - @vocab(dict):
        - @merges(dict): dictionary of merged pairs. if tok1 and tok2 is merged to new_tok, 
          then merges[new_tok] = (tok1, tok2)

        """
        
        assert vocab_size >= 256

        ## preprocess the text
        bytes_txt = text.encode('utf-8')
        ids = list(map(int, bytes_txt))
        
        ## merge and record the merge pairs in the self.merges dictionary
        #  1. get the frequency stats from the corpus
        #  2. find the most frequent pair
        #  3. execute the merge
        #  4. record the merged pairs in the self.merge dictionary
        
        num_merge = 0
        while num_merge < vocab_size - 256:
            # 1) get stats
            cnts = get_stats(ids)

            # 2) get the most frequent pair
            candi_pair = max(cnts, key=cnts.get)
            
            # 3) execute merge
            new_idx = 256 + num_merge
            ids = merge_pair(ids, candi_pair, new_idx)

            # 4) record the merge in self.merges
            self.merges[candi_pair] = new_idx

            # 5) update num_merge
            num_merge += 1

            # 6) (optional)print info. for debug
            if verbose:
                print(f'merge {num_merge}/{vocab_size-256}: {candi_pair} => {new_idx}')

        ## construct self.vocab
        #  self.vocab is the "index -> raw bytes" table represented in an dictionary
        #  - in self.merges: key : value <> (merged_idx_1, merged_idx_2) : new_idx
        #  - in self.vocab:  key : value <> idx : raw_bytes_of_all_ids_merged_in_idx
        self.vocab = self._build_vocab()

        
    def encode(self, text_in):
        """
        mission: convert the str string type input corpus to a list of integers called tokens.
                 the value of each integer(token) is in the range of 0~255.
                 Then the tokens would be processed by a language model.

        inputs:
        @text_in(str string): str string type data. every character is first converted to a 
                              byte string, which then would be split to an integer list with 
                              each integer correcpondes to one byte data in the raw byte string.

        output:
        @tok_in(list of intergers): The value of each element is in the range of 0~vocab_size.
                                    And each element is one token of input that would be 
                                    processed by a language model.
        """
        tok_in = self.encode_single_chunk(text_in)

        return tok_in
    
    def decode(self, toks_out):
        """
        mission: convert the outputs of LM @toks_out from token list to str string.
        
        input:
        @toks_out(list of integers): It is the output token list of LM. 
                                     int in the list is in the range of 0~vocab_size
        """
        bytes_out = b''.join(self.vocab[i] for i in toks_out)
        txt_out = bytes_out.decode(encoding='utf-8', errors='replace')
        return txt_out