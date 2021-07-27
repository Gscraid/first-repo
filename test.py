import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
convertation = {'test': transforms.Compose([transforms.Resize((224,224)),
                                    
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5])]),
                   }
loaders = {'test': torch.utils.data.DataLoader(datasets.ImageFolder('data/', 
                                                                            transform=convertation['test']),
                                                        batch_size=1, shuffle=True)}
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.c2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=6272, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=412)
        self.fc4 = nn.Linear(in_features=412, out_features=4)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.c1(x), 3))
        x = F.relu(F.max_pool2d(self.c2(x), 3))
        x = F.relu(F.max_pool2d(self.c3(x), 3))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        

        return x

model = cnn()
model.load_state_dict(torch.load('trained.pth'))
total_test = 0
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
count = 0
for data,target in loaders['test']:
    
    total_correct = 0
    total = 0
    
    output = model(data)
    loss = criterion(output, target)
    max_arg_output = torch.argmax(output, dim=1)
    total_correct += int(torch.sum(max_arg_output == target))
    total += data.shape[0]
    #print('Test accuracy: {:.0%}'.format(total_correct/total))
    total_test += total_correct/total 
    count += 1
print('Total test accuracy: {:.0%}'.format(total_test/count))
