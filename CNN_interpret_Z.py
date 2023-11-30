#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import os
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
starttime = time.time()

#%%
# ==============================================
# Create data loader 
# ==============================================

class ImageDataset(data.Dataset):
    def __init__(self, data_folder, classes):
        self.data_folder = data_folder
        self.filenames = []
        self.labels = []

        per_class_paths = os.path.join(data_folder, classes)
        label = torch.tensor(int(classes))

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

def choose_validation(size, classes):
    ### Draw validation data of size[size] from dataset
    valid_dataset = ImageDataset(data_folder='data/EXP2/Z/', classes=classes)
    dataset_size = len(valid_dataset)
    indices = list(range(dataset_size))
    shuffle_dataset = True
    if shuffle_dataset:
        np.random.seed(42)
        np.random.shuffle(indices)
    valid_indices = indices[:size]
    valid_sampler = data.sampler.SubsetRandomSampler(valid_indices)
    validation_loader = data.DataLoader(valid_dataset, batch_size=1, sampler=valid_sampler)
    
    return validation_loader


# ==============================================
# Define a convolutional neural network
# ==============================================

def conv_dim(w, ks, p, st, d=1):
    # Calculate the shape after convolution
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

def test(model, dataloader):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for data, label in dataloader:
            data, label = next(iter(dataloader))
            # add channel dimension ([: : :] to [:,1,:,:])
            data = data[:,None,:,:]
            output = model(data)
            # reduce dimension ([:,1] to [:])
            output = torch.squeeze(output)
    return output

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    accuracy = (y_true == predicted).sum().float()/len(y_true)

    return accuracy

def assign_zeros_and_test(model, dataloader, zero_len, X, Y):
    # size of mask should be odd number
    # add padding according to the size of mask
    padding_size = int(zero_len/2)

    model.eval()

    with torch.no_grad():
        running_loss = []
        running_output = []
        running_accuracy = []
        for data, label in dataloader:
            # ------- Get test data & assign zeros -------
            data, label = next(iter(dataloader))
            data, label = data.to(device), label.to(device)
            expanded_size = data.shape[-1] + padding_size*2
            expanded_input = np.zeros((data.shape[0], expanded_size, expanded_size))
            expanded_input[:,padding_size:-padding_size, padding_size:-padding_size] = data.cpu()
            # assign zeros
            expanded_input[:, X:X+zero_len, Y:Y+zero_len] = 0.
            # remove padding
            data_0 = expanded_input[:, padding_size:-padding_size, padding_size:-padding_size]

            # ------- Test -------
            # add input channel dimension ([: : :] to [:,1,:,:])
            data_0 = torch.from_numpy(data_0[:,None,:,:])
            data_0 = data_0.float()
            output = model(data_0)
            # reduce output dimension ([:,1] to [:])
            output = torch.squeeze(output, 1)
            loss = F.binary_cross_entropy(output.float().to(device), label.float().to(device))
            acc = calculate_accuracy(label.float().to(device), output.float().to(device))

            running_loss.append(loss.item())
            running_accuracy.append(acc)
            running_output.append(output)

    return running_loss, running_accuracy, running_output

#%%
# ==============================================
# Validate
# ==============================================
# ========== Load Pretrained Model ==========
path_model = 'models/CNN_statedict_trainwithZ.pt'
model = CNN()
model.load_state_dict(torch.load(path_model))
model.eval()

# ========== Assign 0 and Validate ==========
size = 100     # draw 100 data at a time
zeros_len = 3  # side length of mask in grid points [3/5/7/9/11/13]
times = 10     # times of validation (size*times = total number of tests)
EPOCHS = 5

valid_pred_pertime_SSW = np.zeros((size, times))  # the prediction
valid_pred_pertime_nSSW = np.zeros((size, times))
error_sum_XY_SSW = np.zeros((36, 36))  # error number in each grid
error_sum_XY_nSSW = np.zeros((36, 36)) 

for X in range(36):
    for Y in range(36):
        print('Tries: {}'.format(times+1), 'X:{}, Y:{}'.format(X, Y))
        for t in times:
            print('Validation, times={}'.format(times+1))
            validation_loader_SSW = choose_validation(size=size, classes='1')
            validation_loader_nSSW = choose_validation(size=size, classes='0')

            for epoch in range(EPOCHS):
                valid_loss_SSW, valid_acc_SSW, valid_pred_SSW = assign_zeros_and_test(\
                        model, validation_loader_SSW, zeros_len, X=X, Y=Y)
                valid_loss_nSSW, valid_acc_nSSW, valid_pred_nSSW = assign_zeros_and_test(\
                        model, validation_loader_nSSW, zeros_len, X=X, Y=Y)
                
            valid_pred_pertime_SSW[:,t] = valid_pred_SSW
            valid_pred_pertime_nSSW[:,t] = valid_pred_nSSW
            
        # How many error predictions are there in (size*times) tests -
        error_sum_SSW = (valid_pred_pertime_SSW < 0.5).sum()
        error_sum_nSSW = (valid_pred_pertime_nSSW > 0.5).sum()
        # - per grid
        error_sum_XY_SSW[X, Y] = error_sum_SSW
        error_sum_XY_nSSW[X, Y] = error_sum_nSSW

        print('Error classification:')
        print('ALL SSW: {}'.format(error_sum_SSW))
        print('ALL non-SSW: {}'.format(error_sum_nSSW))

np.savez('models/validation_result_Z/CNN_interpret_masksize{}.npz'.format(zeros_len),\
         error_sum_XY_SSW=error_sum_XY_SSW, error_sum_XY_nSSW = error_sum_XY_nSSW,\
            num_of_tests=size*times)

print('Finished testing!')
endtime = time.time()
convert = time.strftime("%Hh %Mm %Ss", time.gmtime(endtime-starttime))
print('Time used: ' + convert)
