import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from momo import Momo, MomoAdam

# Example neural network
class Net(nn.Module):
    def __init__(self, d, H):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, H)  
        self.fc2 = nn.Linear(H, 1) 

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Define a synthetic dataset    
d = 2   # input dimension
N = 1000  # number of samples
H = 100   # hidden layer size

X = torch.randn(N, d) 
y = (X**2).sum(axis=1)

ds = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
dl = DataLoader(ds, batch_size=10)

# Define the loss function
criterion = nn.MSELoss()

def loss_fn(output, labels):
  loss = criterion(output.view(-1), labels.view(-1))
  loss.backward()
  return loss
  
def train(model, opt, epochs=100):
  for epoch in range(epochs):
    for input, labels in dl:
          opt.zero_grad()
          output = model(input)
          closure = lambda: loss_fn(output, labels) # define a closure that return loss
          
          opt.step(closure=closure)
          # alternative:
          # loss = closure()
          # opt.step(loss=loss)

    # print progress
    if epoch % 10 == 0:
      print('Epoch {}, loss: {}'.format(epoch, 1/N*((model(X).view(-1) - y)**2).sum()))

  return


if __name__ == '__main__':
   model = Net(d,H)
   opt = Momo(model.parameters(), lr=1)
   # opt = MomoAdam(model.parameters(), lr=1e-2)
   train(model, opt, epochs=100)
