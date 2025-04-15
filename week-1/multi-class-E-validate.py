from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch 
import csv
import gensim
import numpy as np
from sklearn.metrics import classification_report

class MyData(Dataset):
  def __init__(self, data, label):
      self.data = torch.tensor(np.array(data), dtype=torch.float32)
      self.label = torch.tensor(label, dtype=torch.long)

  def __getitem__(self, index):
      return self.data[index],self.label[index]

  def __len__(self):
      return len(self.label)

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

def train(dataloader, model, loss_fn, optimizer):
  model.train()
  total_loss = 0
  num_batches = len(dataloader)

  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    total_loss += loss.item()

  avg_loss = total_loss / num_batches
  print(f"Average Loss in Training Data: {avg_loss:.4f}")     

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    correct = 0
    correct_top2 = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for X, y in dataloader:
          X, y = X.to(device), y.to(device)
          pred = model(X)
          
          pred_class = torch.argmax(pred, dim=1)
          correct += (pred_class.eq(y).sum().item())
          
          top2_pred = torch.topk(pred, 2, dim=1).indices
          match_top2 = (top2_pred[:, 0] == y) | (top2_pred[:, 1] == y)
          correct_top2 += match_top2.sum().item()    
        
          test_loss += loss_fn(pred, y).item()

          all_preds.extend(pred_class.cpu().numpy())
          all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    print(f"Average Loss: {test_loss:.4f}")
    correct_rate = correct / size
    correct_rate_top2 = correct_top2 / size
    return  correct_rate, correct_rate_top2, all_preds, all_labels

# Prepare Data
xs = []
es = []
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

doc2vec_model = gensim.models.Doc2Vec.load("doc2vec_model_4.bin")

with open("data-clean-words.csv", mode='r', newline='', encoding='utf-8-sig') as read_file:
  reader = csv.reader(read_file)
  text = list(reader)
  for i in range(0, len(text)):
      tokens = []
      for j in range(1, len(text[i])):
        if len(text[i][j]) > 0:
          tokens.append(text[i][j])
      if len(tokens) > 0:
        xs.append(doc2vec_model.infer_vector(tokens))
        es.append(boards[text[i][0].lower()])

from collections import Counter
print("Label 分布統計：")
print(Counter(es))

input_dim = len(xs[0])
output_dim = len(boards)

print("------ Before Training ------")
dataset = MyData(xs, es)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - val_size -train_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

model = NeuralNetwork(input_dim, output_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

correct_rate, correct_rate_top2, all_preds, all_labels = test(test_loader, model, loss_fn)
print("First Match", correct_rate*100, "%")
print("Second Match", correct_rate_top2*100, "%")
print(classification_report(all_labels, all_preds, digits=4))

print("------ Start Training ------")

epochs = 50
for t in range(epochs):
  print(f"Epoch {t+1}\n------------")
  train(train_loader, model, loss_fn, optimizer)

  print("[Train Set]")
  train_acc, train_top2, train_preds, train_labels = test(train_loader, model, loss_fn)
  print("First Match", train_acc*100, "%")
  print("Second Match", train_top2*100, "%")
  print(classification_report(train_labels, train_preds, digits=4))

  print("[Validate Set]")
  correct_rate, correct_rate_top2, all_preds, all_labels = test(val_loader, model, loss_fn)
  print("First Match", correct_rate*100, "%")
  print("Second Match", correct_rate_top2*100, "%")
  print(classification_report(all_labels, all_preds, digits=4))

torch.save(model, 'model_full_val.pth')
torch.save(model.state_dict(), 'model_state_dict_val.pth')

print("[Test Set]")
correct_rate, correct_rate_top2, all_preds, all_labels = test(test_loader, model, loss_fn)
print("First Match", correct_rate*100, "%")
print("Second Match", correct_rate_top2*100, "%")
print(classification_report(all_labels, all_preds, digits=4))

