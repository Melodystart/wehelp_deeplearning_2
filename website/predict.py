import torch
from torch import nn

boards = {
  'Baseball':0, 
  'Boy-Girl':1, 
  'C_Chat':2, 
  'HatePolitics':3, 
  'Lifeismoney':4, 
  'Military':5,
  'PC_Shopping':6, 
  'Stock':7, 
  'Tech_Job':8
}

def prediction(title, ws_driver, pos_driver, doc2vec_model, model):

  # tokenizer
  text_tokens = []

  filter_pos = ["Caa", "Cab", "Cba", "Cbb", "P", "DE", "FW", "COLONCATEGORY","COMMACATEGORY", "DASHCATEGORY", "ETCCATEGORY", "EXCLANATIONCATEGORY", "PARENTHESISCATEGORY", "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY", "PARENTHESISCATEGORY", "WHITESPACE", "EXCLAMATIONCATEGORY"]

  ws  = ws_driver([title])
  pos = pos_driver(ws)

  for word_ws, word_pos in zip(ws[0], pos[0]):
      if word_pos not in filter_pos:
        text_tokens.append(word_ws)

  # classification
  device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"
  inferred_vector = doc2vec_model.infer_vector(text_tokens)
  input_tensor = torch.from_numpy(inferred_vector).float().unsqueeze(0).to(device)

  index_to_board = {v: k for k, v in boards.items()}
  with torch.no_grad():
    pred = model(input_tensor)
    prob = torch.softmax(pred, dim=1)   

  result = {}
  for i, prob in enumerate(prob[0]):
    result[index_to_board[i]] = round(prob.item(), 4)
  sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
  return sorted_result, text_tokens
