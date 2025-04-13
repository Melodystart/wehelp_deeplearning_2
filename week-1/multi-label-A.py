from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch 
import csv
import gensim
import numpy as np

class MyData(Dataset):
  def __init__(self, data, label):
      self.data = torch.tensor(np.array(data), dtype=torch.float32)
      self.label = torch.tensor(label, dtype=torch.float32)

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
    
    with torch.no_grad():
      for X, y in dataloader:
          X, y = X.to(device), y.to(device)
          pred = model(X)
          prob = torch.sigmoid(pred)

          pred_labels = (prob >= 0.5).int()
          true_labels = y.int()
          for i in range(len(true_labels)):
              if (pred_labels[i] & true_labels[i]).sum().item() > 0:
                  correct += 1

          test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Average Loss: {test_loss:.4f}")
    correct_rate = correct / size
    return  correct_rate

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
num_labels = len(boards)

doc2vec_model = gensim.models.Doc2Vec.load("doc2vec_model.bin")

with open("data-clean-words-sample.csv", mode='r', newline='', encoding='utf-8-sig') as read_file:
  reader = csv.reader(read_file)
  text = list(reader)
  for i in range(0, len(text)):
      tokens = []
      for j in range(1, len(text[i])):
        if len(text[i][j]) > 0:
          tokens.append(text[i][j])
      if len(tokens) > 0:
        xs.append(doc2vec_model.infer_vector(tokens))

        label_vec = [0] * num_labels
        label_name = text[i][0].lower()
        label_vec[boards[label_name]] = 1
        es.append(label_vec)

input_dim = len(xs[0])
output_dim = len(boards)

print("------ Before Training ------")
dataset = MyData(xs, es)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.accelerator.current_accelerator().type if torch.cuda.is_available() else "cpu"

model = NeuralNetwork(input_dim, output_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

correct_rate = test(test_loader, model, loss_fn)
print("Accuracy", correct_rate*100, "%")

print("------ Start Training ------")

epochs = 200
for t in range(epochs):
  print(f"Epoch {t+1}\n------------")
  train(train_loader, model, loss_fn, optimizer)
  correct_rate = test(test_loader, model, loss_fn)
  print("Accuracy", correct_rate*100, "%")
  

