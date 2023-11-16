#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
starttime = time.time()

# =================================================
# Create data loader and split into train/validate
# =================================================

class ImageDataset(data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.filenames = []
        self.labels = []

        per_classes = os.listdir(data_folder)
        for per_class in per_classes:
            per_class_paths = os.path.join(data_folder, per_class)
            label = torch.tensor(int(per_class))

            per_datas = os.listdir(per_class_paths)
            for per_data in per_datas:
                self.filenames.append(os.path.join(per_class_paths, per_data))
                self.labels.append(label)

    def __getitem__(self, index):
        # Generates one sample of data #
        data = np.load(self.filenames[index], allow_pickle=True)
        data = torch.from_numpy(data)
        label = self.labels[index]
        return data, label
    
    def __len__(self):
        # Denotes the total number of samples #
        return len(self.filenames)

my_dataset = ImageDataset(data_folder='data/EXP1/U/')
batch_size = 16
validation_split = 0.3
shuffle_dataset = True
random_seed = 42

# Create data indices for training and validation splits:
dataset_size = len(my_dataset)  # for EXP1: 759 * 2 = 1518
indices = list(range(dataset_size))
split = int(np.floor(validation_split  * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = data.DataLoader(my_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = data.DataLoader(my_dataset, batch_size=batch_size, sampler=test_sampler)
print('there are total %s batches for train' % (len(train_loader)))
print('there are total %s batches for validation' % (len(test_loader)))

# ==============================================
# Define a convolutional neural network
# ==============================================

def conv_dim(w, ks, p, st, d=1):
    # calculate the shape after convolution
    # w:input dim., ks:kernel size, p:padding, s:stride, d:dilation 
    return int((w + 2*p - d*(ks-1) - 1)/st + 1)

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        nw = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(36,3,1,1),3,1,1),2,0,2),3,1,1),3,1,1)
        self.fc1 = nn.Linear(16*nw*nw, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):

         nw = conv_dim(conv_dim(conv_dim(conv_dim(conv_dim(36,3,1,1),3,1,1),2,0,2),3,1,1),3,1,1)
         x = F.relu(self.bn1(self.conv1(x)))
         x = F.relu(self.bn2(self.conv2(x)))
         x = self.pool(x)
         x = F.relu(self.bn4(self.conv4(x)))
         x = F.relu(self.bn5(self.conv5(x)))
         x = x.view(-1, 16*nw*nw)
         x = F.relu(self.fc1(x))
         x = torch.sigmoid(self.fc2(x))

         return x

def train(model, optimizer):
    model.train()
    model.to(device)
    #train_loss, train_acc = [], []
    #for epoch in range(EPOCHS):
    running_loss = []
    running_acc = []
    for i, (data, label) in enumerate(train_loader):
        data = Variable(data.to(device))
        label = Variable(label.to(device))
        # add channel dimension ([: : :] to [:,1,:,:])
        data = data[:,None,:,:]
        # zero the parameter gradients
        optimizer.zero_grad()
        output = model(data)
        # reduce dimension ([:,1] to [:])
        output = torch.squeeze(output)
        loss = F.binary_cross_entropy(output.to(torch.float32), label.to(torch.float32))
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(label, output)
        running_loss.append(loss.item())
        running_acc.append(acc)
            # print statistics 
        if i % 10 == 0:
            print('batch {} [{}/{} ({:.2f}%)]\tLoss: {:.6f} accuracy:{}'\
                .format(i, i*len(data), dataset_size-split, \
                    100*i/len(train_loader), loss.item(), acc))

    # Compute and print the average loss and acc for this epoch
    avg_loss, avg_acc = np.mean(running_loss), np.mean(running_acc)

    return avg_loss, avg_acc###

def test(model):
    model.eval()
    with torch.no_grad():
        running_loss = []
        running_acc = []
        for data, label in test_loader:
            data, label = next(iter(test_loader))
            # add channel dimension ([: : :] to [:,1,:,:])
            data = data[:,None,:,:]
            output = model(data)
            # reduce dimension ([:,1] to [:])
            output = torch.squeeze(output)
            loss = F.binary_cross_entropy(output.to(torch.float32), label.to(torch.float32))
            acc = calculate_accuracy(label, output)

            running_loss.append(loss.item())
            running_acc.append(acc)
    test_loss, test_acc = np.mean(running_loss), np.mean(running_acc)
    print('\nTest set: Average loss: {:.4f}, accuracy: {}'\
        .format(test_loss, test_acc))
    #print(np.shape(running_acc))# 29
    return test_loss, test_acc

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    accuracy = (y_true == predicted).sum().float()/len(y_true)

    return accuracy

def saveModel():
    # save the parameters of trained model
    path = 'models/'
    torch.save(model.state_dict(), path+'CNN_statedict_trainwithU.pt')
    print('Model state dict saved')


#%%
# ======== Model ========
model = CNN()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.002)

EPOCHS = 400
train_loss_perEP, train_acc_perEP = [], [] # per epoch
test_loss_perEP, test_acc_perEP = [], []
for epochs in range(EPOCHS):
    print('\nEPOCH {}:'.format(epochs + 1))
    avg_loss, avg_acc = train(model, optimizer)
    train_loss_perEP.append(avg_loss), train_acc_perEP.append(avg_acc)
    test_loss, test_acc = test(model)
    test_loss_perEP.append(test_loss), test_acc_perEP.append(test_acc)
print('Finish training & validating')
saveModel()

endtime = time.time()
convert = time.strftime("%Hh %Mm %Ss", time.gmtime(endtime-starttime))
print('Time used: ' + convert)


# Plot loss & accuracy with epoch
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11,6))
ax[0].plot(np.arange(len(train_loss_perEP))+1, train_loss_perEP, label='Train loss', color='k')
ax[0].plot(np.arange(len(test_loss_perEP))+1, test_loss_perEP, label = 'Test loss', color='tab:red')
ax[0].legend()
ax[0].set_title('Loss')
ax[0].set_xlabel('epochs')

ax[1].plot(np.arange(len(train_acc_perEP))+1, train_acc_perEP, label='Train accuracy', color='k')
ax[1].plot(np.arange(len(test_acc_perEP))+1, test_acc_perEP, label = 'Test accuracy', color='tab:red')
ax[1].legend()
ax[1].set_xlabel('epochs')
ax[1].set_title('Accuracy')
plt.savefig('figures/CNN_lossnacc_U.png', dpi=250)
