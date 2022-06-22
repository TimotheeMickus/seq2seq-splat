# Seq2seq SPLAT

This is an extension of the [TACL paper decomposition](https://arxiv.org/pdf/2206.03529.pdf)
more specifically for sequence-to-sequence (MT) transformers.

## Get started
The file `extract_marianmt.py` defines a `Decomposer` class. That's your main
entrypoint. Construct a `Decomposer`, then call it with the source and target
sentences. That will return a  `L x T x 5 x D` tensor where:
+ `L` is either 1 or the number of layers in the model,
+ `T` is the target sequence length in tokens,
+ 5 corresponds to the terms we decompose embeddings into, and
+ `D` is the embedding size of the model.

Note that the file also defines some short hands for decomposition indices (`I`,
`S`, `T`, `F`, `C`)

An example of usage is provided in `dump_sims.py`, which defines a few
similarity metrics and dumps in a CSV all similarity measurements for an input
parallel corpus.

## Stuff to do now
+ Test models across training
+ Test more rigourously across languages
+ Add automatic quality metrics (e.g. BLEU) and other useful explanation metrics (e.g. LRP)
+ Check whether low source importance entails hallucinations
