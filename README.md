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

Similar scripts for token-wise decomposition inspired by Oh & Schuler (2023) can also be found in this repo.

## Citation

If this code has been useful to you in any way, please consider citing our publication:
```bibtex
@inproceedings{mickus-vazquez-2023-bother,
    title = "Why Bother with Geometry? On the Relevance of Linear Decompositions of Transformer Embeddings",
    author = "Mickus, Timothee  and
      V{\'a}zquez, Ra{\'u}l",
    editor = "Belinkov, Yonatan  and
      Hao, Sophie  and
      Jumelet, Jaap  and
      Kim, Najoung  and
      McCarthy, Arya  and
      Mohebbi, Hosein",
    booktitle = "Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.blackboxnlp-1.10",
    doi = "10.18653/v1/2023.blackboxnlp-1.10",
    pages = "127--141",
    abstract = "A recent body of work has demonstrated that Transformer embeddings can be linearly decomposed into well-defined sums of factors, that can in turn be related to specific network inputs or components. There is however still a dearth of work studying whether these mathematical reformulations are empirically meaningful. In the present work, we study representations from machine-translation decoders using two of such embedding decomposition methods. Our results indicate that, while decomposition-derived indicators effectively correlate with model performance, variation across different runs suggests a more nuanced take on this question. The high variability of our measurements indicate that geometry reflects model-specific characteristics more than it does sentence-specific computations, and that similar training conditions do not guarantee similar vector spaces.",
}
```
