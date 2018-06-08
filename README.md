# word2vec
Similarity measure with corpus of art reviews - "art thesaurus"


We took as our corpus of words (i.e. the possible outputs of the thesaurus) a relatively small number of art reviews found at various reputable sources. Then we gave the model a “target” word; this is the word we want see context for. Each word can be mapped to a vector by the GLoVE algorithm and then we can use some maths to find the 5 most “similar” or “closest” words in the corpus to the target word.
