# Seq2seq SPLAT

This is the code repository for our BlackBoxNLP 2023 paper "_Why bother with geometry? On the relevance of linear decompositions of Transformer embeddings._"
It mainly contains utility classes to linearly decompose input sentences, along with some analysis scripts and result files.

## Use this code
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

Similar scripts for token-wise decomposition as proposed by Oh & Schuler, 2023 can also be found in the
