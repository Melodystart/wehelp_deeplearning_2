import logging
import csv
import gensim
import collections
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_corpus(csv_file, tokens_only=False):
  with open(csv_file, mode='r', newline='', encoding='utf-8-sig') as read_file:
    reader = csv.reader(read_file)
    text = list(reader)
    for i in range(0, len(text)):
        tokens = []
        for j in range(1, len(text[i])):
          if len(text[i][j]) > 0:
            tokens.append(text[i][j])
        if len(tokens) > 0:
          if tokens_only:
              yield tokens
          else:
              # For training data, add tags
              yield gensim.models.doc2vec.TaggedDocument(tokens, [text[i][0]])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_1 = os.path.join(BASE_DIR, "data-clean-words.csv")
csv_2 = os.path.join(BASE_DIR, "user-labeled-words.csv")

train_corpus = list(read_corpus(csv_1)) + list(read_corpus(csv_2))

# test_corpus = list(load_corpus_from_csv("test_tokens.csv", tokens_only=True))
model = gensim.models.doc2vec.Doc2Vec(vector_size=50,  window=5, min_count=2, epochs=100,  dm=0, dbow_words=1, workers=4)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save("doc2vec_model_4.bin")

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [tag for tag, sim in sims].index(train_corpus[doc_id].tags[0])
    ranks.append(rank)

    second_ranks.append(sims[1])
    
counter = collections.Counter(ranks)
print(counter)
print(f"vector_size=50,  window=5, min_count=2, epochs={model.epochs}, dm=0, dbow_words=1, workers=4")
print(f"Self-Similarity: {counter[0] / len(ranks) * 100:.2f}%")
print(f"Second Self-Similarity: {(counter[0]+counter[1]) / len(ranks) * 100:.2f}%")
