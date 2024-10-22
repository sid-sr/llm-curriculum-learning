# Curriculum Learning on Language Models
<h4>[Report](Curriculum_Learning_Report.pdf) </h4>
<h3>Notebooks in the "scripts" folder:</h3>
1. perplexity_lstm_training.ipynb: Based on https://arxiv.org/pdf/2110.02406.pdf, trains a BILSTM to generate surprisals for each word at fixed checkpoints. (Run on Colab). [Author's code](https://github.com/tylerachang/word-acquisition-language-models/).
2. wordbank_bilstm_aoa_analysis.ipynb: Also based on https://arxiv.org/pdf/2110.02406.pdf, compares their provided LSTM AoAs with AO-CHILDES ground truth AoA.
3. portelance_aoa_analysis.ipynb: Based on https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.13334, uses their LSTM (and unigram, bigram etc.) surprisals as predictors for AoA (obtained for AO-CHILDES words). Does not train language models, we already have the result files from their code [Author's code](https://github.com/evaportelance/multilingual-aoa-prediction).

<h1>Curriculum Development Using Age of Acquisition and Surprisal:</h1>

<h3>Abstract</h3>
The Age of Acquisition (AoA) of a word refers to an estimate of the age at which children produce the word with a probability greater than some threshold, i.e., the age at which children learn a particular word. Existing databases of children’s vocabulary development, such as Stanford’s Wordbank repository (Frank et al., 2016), only contain words with AoAs ranging from 16 to 30 months. By training language models on child-directed speech, researchers have shown that a model’s average surprisal for a given word aids in predicting AoA of that word beyond traditional predictors (Portelance et al., 2023). Our project trains language models (LMs) on AO-CHILDES (Huebner & Willits, 2021), a child-directed speech dataset, to model any word’s AoA
using its average surprisal, extending beyond the words present in Wordbank. We then build a curriculum (Bengio et al., 2009) based on increasing AoA to train LMs such as BabyBERTa (Huebner et al., 2021) with the ordering of words mirroring a child learning English. The AO-CHILDES dataset also provides a ground truth curriculum to use.

<h3> Methodology </h3>

<h4> AoA Prediction </h4>
To predict AoA as a function of model surprisal, we first
train a language model on AO-CHILDES to obtain the
average surprisal of a word over all its occurrences in the
corpus. Let 𝐶 be the set of documents in the corpus that a
token 𝑣 occurs in. For a language model characterized by
the probability distribution 𝑃 over tokens, the surprisal of 𝑣
is given by:
𝑠𝑢𝑟𝑝(𝑣) =
1
|𝐶|
𝑤 ∈ 𝐶; 𝑤𝑖= 𝑣
∑ − log 𝑃(𝑤
𝑖
| 𝑤
𝑖−1
, 𝑤
𝑖−2
.. 𝑤1
)
where 𝑤 is a document (sequence of tokens) in 𝐶.
We used two methods to obtain the AoA value based on
surprisal:
- Final AoA: Extract the average surprisal values
over the corpus for each token on the final trained
model.
- Midpoint AoA: The method used by Chang et al.
(2022), where we track the development of average
surprisal for each token as the model gets trained.
We then find the training step count at which the
surprisal reaches halfway between a random
sampling baseline (no training) and its minimum
value once trained.
The final AoA method uses the surprisal value as the AoA
scale, while the midpoint AoA uses step count as the scale.
Similar to Chang et al. (2022), we trained a bidirectional
LSTM model from scratch to obtain surprisals. We also
tested with Pythia (Biderman et al., 2023), a pre-trained
large language model (LLM) to get surprisal values. We
used the 14M-parameter, 6-layer configuration of the model.
Curriculum Building
After predicting the AoA values for each word and storing
them, the next step was to build a curriculum by ordering
the sentences of AO-CHILDES. The challenge was to order
the sentences ensuring that the sentences containing words
with lower AoA values would appear at the start of the
ordering, and sentences containing words with higher AoA
values would appear at the end of the ordering. To achieve
this, we implemented three strategies:
- Sentence AoA based on token with maximum
AoA.
- Sentence AoA based on average AoA of tokens in
the sentence (without removing any stopwords).
- Sentence AoA based on average AoA of tokens in
the sentence (removing stopwords with NLTK).
The first strategy was quite straightforward and prioritizes
the most complex word in the sentence. The sentence AoA
is simply the AoA of the word with the highest AoA value
within the sentence. This strategy assumes that mastering
the most challenging word will facilitate understanding the
entire sentence, but ignoring the AoA of the other words in
the sentence may lead to an incorrect estimate of the overall
difficulty.
In the second strategy, the AoA of a sentence is calculated
as the average AoA of all its words. This method takes into
account the contribution of all the words in the sentence to
the sentence’s overall complexity. However, this strategy
may result in an undesired reduction of sentence AoA as
stopwords (“the”, “is”, “and”, etc.) would likely have lower
AoA value than regular words.
The third strategy is similar to the second strategy, but it
excludes stopwords. Stopwords are words that are filtered
out before processing natural language data, and NLTK
provides a list of common English stopwords. This strategy
thus prioritizes words in the sentence that carry more
meaning.
