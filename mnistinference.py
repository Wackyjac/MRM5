import torch
import torchvision  
import torch.nn.functional as F
from mnistmodelfinal import Net
import warnings
from sklearn.metrics import f1_score, confusion_matrix


warnings.filterwarnings("ignore", category=UserWarning)
batch_size_test = 1000
random_seed=1
torch.manual_seed(random_seed)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=batch_size_test, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = torch.cat([data for data, _ in test_loader], dim=0)
mean_value = test_data.mean()
std_value = test_data.std()
print(mean_value)
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mean_value,), (std_value,))
])
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=transform),
  batch_size=batch_size_test, shuffle=True)




network = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)
model_path = 'C:\\Shaurya\\results\\model.pth'
network_state_dict = torch.load(model_path)
network.load_state_dict(network_state_dict)

labels=[]
final_preds=[]
def test(loader):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      data=data.to(device)
      target=target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(loader.dataset),
    100. * correct / len(loader.dataset)))
  final_preds.extend(pred.cpu().numpy())
  labels.extend(target.cpu().numpy())
  print(data.shape)
  return final_preds,labels

def testlive(img):
  network.eval()
  with torch.no_grad():
    img1 = img/255.
    data = img1.to(device)
    output=network(data)
    pred = output.data.max(1, keepdim=True)[1]
  print(pred)

if __name__=='__main__':
  pred,target=test(test_loader)

  print(f1_score(pred, target, average='macro'))
  print(f1_score(pred, target, average='micro'))
  print(confusion_matrix(target,pred))