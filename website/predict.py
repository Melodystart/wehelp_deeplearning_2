import torch
from torch import nn
import gensim
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

class NeuralNetwork(nn.Module):
  def __init__(self, input_dim, output_dim):
      super().__init__()
      self.stack = nn.Sequential(
        nn.Linear(input_dim, 32),  
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, output_dim)
      )
  def forward(self, x):
      logits = self.stack(x)
      return logits

boards = {
  'baseball':0, 
  'boy-girl':1, 
  'c_chat':2, 
  'hatepolitics':3, 
  'lifeismoney':4, 
  'military':5,
  'pc_shopping':6, 
  'stock':7, 
  'tech_job':8
}

def prediction(title):

  # tokenizer
  text_tokens = []

  ws_driver  = CkipWordSegmenter(model="bert-base")
  pos_driver = CkipPosTagger(model="bert-base")

  filter_pos = ["Caa", "Cab", "Cba", "Cbb", "P", "DE", "FW", "COLONCATEGORY","COMMACATEGORY", "DASHCATEGORY", "ETCCATEGORY", "EXCLANATIONCATEGORY", "PARENTHESISCATEGORY", "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY", "PARENTHESISCATEGORY", "WHITESPACE", "EXCLAMATIONCATEGORY"]

  ws  = ws_driver([title])
  pos = pos_driver(ws)

  for word_ws, word_pos in zip(ws[0], pos[0]):
      if word_pos not in filter_pos:
        text_tokens.append(word_ws)

  # print(text_tokens)

  # classification
  device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
  doc2vec_model = gensim.models.Doc2Vec.load("doc2vec_model_4.bin")
  inferred_vector = doc2vec_model.infer_vector(text_tokens)
  input_tensor = torch.from_numpy(inferred_vector).float().unsqueeze(0).to(device)

  index_to_board = {v: k for k, v in boards.items()}

  input_dim = doc2vec_model.vector_size
  output_dim = len(boards)


  model = NeuralNetwork(input_dim, output_dim).to(device)
  model.load_state_dict(torch.load("model_state_dict.pth"))
  model.eval()

  with torch.no_grad():
    pred = model(input_tensor)
    prob = torch.softmax(pred, dim=1)   

  result = {}
  for i, prob in enumerate(prob[0]):
    result[index_to_board[i]] = round(prob.item(), 4)
  sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
  return sorted_result
