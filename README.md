# rebuildminBPE
Rebuild the minBPE tokenizer following the guidance from Andrey Karpathy's YouTube lecture.

There are four python files:
1. 'base.py' is the base class for unifying the interface and has some helper functions.
2. 'basic_tokenizer.py' is a simple byte-level BPE tokenizer implemeted the BPE algorithm.
3. 'regex_tokenizer.py' is able to handle an optional regex splitting pattern. and also accommadate optional special tokens.
4. 'rebuild_gpt4.py' reocovered the merge pairs and the vocabulary table of the GPT4 tokenizer. Then used them as a pre-trained model the same way as in regex_tokenizer.

There are some differences from Andrey's original implementation:
1. More comments have been added to explain the implementation detail. 
2. Use a simpler way to handle the shuffled ranks in the GPT4 vocabulary table. This allows the recovered tokenizer to save and load the same way as in regex_tokenizer, which are not allowed in the original implementation.
