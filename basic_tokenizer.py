class BasicTokenizer():
    def __init__(self):
        self.vocab = []
        self.merges = {}
    
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

        ## preprocess the text
        bytes_txt = text.encode('utf-8')
        ids_txt = list(map(int, bytes_txt))
        
        ## merge and record the merge pairs in the self.merges dictionary
        #  1. get the frequency stats from the corpus
        #  2. find the most frequent pair
        #  3. execute the merge
        #  4. record the merged pairs in the self.merge dictionary
        
        ids = ids_txt[:]
        num_merge = 0
        while num_merge < vocab_size - 256:
            #  get stats
            cnts = self.get_stats_(ids)

            #  get the most frequent pair
            candi_pair = max(cnts, key=cnts.get)
            
            #  execute merge
            ids = self.merge_pair_(ids_txt, candi_pair, vocab_size + num_merge)

            #  record the merge in self.merges
            self.merges[candi_pair] = vocab_size+num_merge
            num_merge += 1

        ## construct self.vocab
        #  self.vocab is the "index -> raw bytes" table represented in an dictionary
        #  it records the information about merged pairs, but unlike the self.merges:
        #  - in self.merges: key : value <> new_idx : (merged_idx_1, merged_idx_2)
        #  - in self.vocab:  key : value <> idx : raw_bytes_of_all_ids_merged_in_this_idx
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        
    
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
        # convert the input text from str string to raw bytes
        bytes_in = text_in.encode('utf-8')
        
        # then split it to a list with each element stands for a byte in the raw bytes data
        idx_in = list(map(int, bytes_in))
        
        # execute the merging
        tok_in = self.merge_all_(idx_in)

        return tok_in
    
    def decode(self, toks_out):
        """
        mission: convert the outputs of LM @toks_out from token list to str string.
        
        input:
        @toks_out(list of integers): It is the output token list of LM. 
                                     each integer in the list is in the range of 0~vocab_size
        
        """
        bytes_out = b''.join(self.vocab[i] for i in toks_out)
        txt_out = bytes_out.decode(encoding='utf-8', errors='replace')
        return txt_out
    
    # help function used in self.encode()
    def merge_all_(self, ids):
        """
        inputs:
        @ids(list of integers): the value of ach element is in the range of 0~255
        @mergers(dictionary of merged pairs)
        
        output:
        @toks(list of integers): list of tokens that would become the inputs of the LM.
                                 The value of each token is in the range of 0~vocab_size. 
        """
        # ## low efficiency way:
        #    The corpus of LM and that of Tokenizer can be different, no need to search every pair
        #    
        # toks = ids[:]
        # for pair, idx in self.merges:
        #     toks = self.merge_pair_(toks, pair, idx)

        ## smarter way:
        
        return toks
    
    # help function used in self.train()
    def get_stats_(self, ids):
        """
        output:
        @cnt({k=pair: v=frequency_count}): frequency count for each pair of consecutive elements in @ids
        """
        cnt = {}
        for pair in zip(ids, ids[1:]):
            cnt[pair] = cnt.get(pair, 0) + 1
        return cnt
    
    # help function used in self.train() and self.merge_all_()
    def merge_pair_(self, ids, pair, idx):
        """
        inputs:
        @ids(list of integers): stands for the token state of the training text before merge.
                                Each pair of its consecutive elements will be matched to the 
                                target pair @pair. Any matched pair will be replaced by the corresponding
                                new index specifiled by @idx.
        @pair(tuple of two integer): target pair of indexes. if any pair of consecutive elements in the 
                                     @ids has the same value of it, they will be replaced.
        @idx(int): a new index that will substitute the @pair in the @ids

        output:
        @new_ids(list of integers): stands for the new token state of the text after every pair of 
                                    consecutive elements that has the same value as the @pair
                                    has been merged.

        """
        new_ids = []
        i = 0
        while i < len(ids) - 1:
            if ids[i] == pair[0] and ids[1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        # deal with the could be left single token, the last element in @ids
        if i < len(ids):
            new_ids.append(ids[i])

        return new_ids
