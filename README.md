# AI-ML-based-text-summarization
An AI/ML based solution to summary solution.

Text is first cleaned with non-alphanumeric chars and stopwords removed.Then splits text into word tokens that are then used to form sentence vectors. The sentence vectors then are used to form a similarity matrix. The ranking in the similarity matrix.  page rank is then used on the sort scores of nodes in the matrix and determine a summary. Sample csv is provided.
