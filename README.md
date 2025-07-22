# AI-ML-based-text-summarization
An AI/ML based solution to summary solution.

Text is first cleaned with non-alphanumeric chars with stopwords removed and put in a list of sentences.The Glove dataset of word vectors is imported with each word and it's corresponding vector space values. The sum of each word vector space values are summed to produce the sentence vector space values,each of which is put into a list. This list is then used to create a similarity matrix, that is then ranked and sorted to determine a summary by sort order. Sample csv is provided.
