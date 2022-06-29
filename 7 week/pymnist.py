
from torch.utils.data import Dataset, DataLoader
import os 
import torchvision
import torchvision.transforms as transfroms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# path = 'C:/Users/lukas/Desktop/Ml 스터디/7 week'
# train_img = os.path.join(path, 'train-images.idx3-ubyte')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
train_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)
test_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = False,
    download = True,
    transform = transfroms.Compose([
        transfroms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)


traindata_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True
                                         )

testdata_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=True
                                         )

dataloaders_dict = {"train" : traindata_loader}
batch_iterator = iter(dataloaders_dict["train"])
inputs, labels = next(batch_iterator)

breakpoint()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(3):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    epoch_loss = 0
    for i, data in enumerate(traindata_loader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    writer.add_scalar('epoch/train', epoch_loss, epoch+1)
            # writer.add_scalar('epoch/train_acc',100*train_acc.count(1)/ dl_len, epoch+1)
        
writer.flush()
     
print('Finished Training')


correct = 0
total = 0
# 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요가 없습니다
with torch.no_grad():
    for data in testdata_loader:
        images, labels = data
        # 신경망에 이미지를 통과시켜 출력을 계산합니다
        outputs = net(images)
        # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        breakpoint()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#tensorboard --logdir=runs