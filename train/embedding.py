import logging
import csv
import gensim
import collections

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

train_corpus = list(read_corpus("data-clean-words-sample.csv")) + list(read_corpus("user-labeled-words-sample.csv"))
# test_corpus = list(load_corpus_from_csv("test_tokens.csv", tokens_only=True))
model = gensim.models.doc2vec.Doc2Vec(vector_size=50,  window=5, min_count=2, epochs=100,  dm=0, dbow_words=1, workers=4)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save("doc2vec_model_sample.bin")

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


# 2025-03-30 10:24:06,160 : INFO : saved doc2vec_model.bin
# Counter({0: 636693, 1: 103523, 2: 48265, 3: 28934, 4: 19817, 5: 14751, 6: 11570, 8: 9969, 7: 9472})
# vector_size=50, min_count=2, epochs=100
# Self-Similarity: 72.11%
# Second Self-Similarity: 83.83%

# 2025-04-10 10:50:07,918 : INFO : saved doc2vec_model_3.bin
# Counter({0: 627473, 1: 111666, 2: 51718, 3: 28761, 4: 18403, 5: 13503, 8: 11634, 6: 10698, 7: 9138})
# vector_size=50,  window=5, min_count=2, epochs=100,  dm=1, dm_mean=1, workers=4
# Self-Similarity: 71.06%
# Second Self-Similarity: 83.71%

# 2025-04-09 11:35:09,819 : INFO : saved doc2vec_model_1.bin
# Counter({0: 642385, 1: 103804, 2: 47025, 3: 27663, 4: 18054, 5: 12956, 8: 11471, 6: 10377, 7: 9259})
# vector_size=50, min_count=2, epochs=100, dm=1, workers=4
# Self-Similarity: 72.75%
# Second Self-Similarity: 84.51%

# 2025-04-09 23:50:18,077 : INFO : saved doc2vec_model_0.bin
# Counter({0: 644370, 1: 120854, 2: 51381, 3: 27549, 4: 16638, 5: 9880, 6: 6242, 7: 3844, 8: 2236})
# vector_size=50, min_count=2, epochs=100, dm=0, dbow_words=1, workers=4
# Self-Similarity: 72.98%
# Second Self-Similarity: 86.66%

# 2025-04-10 22:06:36,603 : INFO : saved doc2vec_model_4.bin
# Counter({0: 652616, 1: 116621, 2: 50728, 3: 26683, 4: 15512, 5: 9534, 6: 5906, 7: 3436, 8: 1958})
# vector_size=50,  window=5, min_count=2, epochs=100, dm=0, dbow_words=1, workers=4
# Self-Similarity: 73.91%
# Second Self-Similarity: 87.12%