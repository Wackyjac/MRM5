import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mnistmodelfinal import Net


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

complete_train = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor()
                                          ]))

train_size = int(0.8 * len(complete_train))
val_size = len(complete_train) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(complete_train, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size_train, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size_train, shuffle=False)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_loader.dataset.transform = transform
val_loader.dataset.transform = transform

examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(3):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
val_losses = []
val_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
    torch.save(network.state_dict(), 'C:\\Shaurya\\results\\model.pth')

def validate():
  network.eval()
  val_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in val_loader:
      data = data.to(device)
      target = target.to(device)
      output = network(data)
      val_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  val_loss /= len(val_loader.dataset)
  val_losses.append(val_loss)
  print('Val set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    val_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))


if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        validate()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(val_counter, val_losses, color='red')
plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show