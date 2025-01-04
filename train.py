#reference https://github.com/neuraloperator/neuraloperator
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("/dssg/home/acct-esehazenet/hazenet-zhanglanyi/fourier-neural-operator-main/")
import matplotlib.pyplot as plt
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from Adam import Adam
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import os
os.makedirs('/dssg/home/acct-esehazenet/hazenet-zhanglanyi/trained_models', exist_ok=True)
os.makedirs('/dssg/home/acct-esehazenet/hazenet-zhanglanyi/saved_models', exist_ok=True)
#################################################################
#def r2
#################################################################
def r2_score(targets, outputs):
    mean_targets = torch.mean(targets)
    ss_1 = torch.sum((targets - outputs)**2)
    ss_2 = torch.sum((targets - mean_targets)**2)
    r2 = 1-ss_1 / ss_2
    return r2.item()
#################################################################
#def loss
#################################################################
def compute_loss(out, y, myloss, batch_size):
    """
    计算加权损失，包括前 37 层和后 7 层的损失。(37层为大气边界层）
    """
    out_front, out_back = out[:, :37, :, :], out[:, 37:, :, :]
    y_front, y_back = y[:, :37, :, :], y[:, 37:, :, :]
    loss_front = myloss(out_front.reshape(batch_size, -1), y_front.reshape(batch_size, -1))  # 前 37 层损失
    loss_back = myloss(out_back.reshape(batch_size, -1), y_back.reshape(batch_size, -1))     # 后 7 层损失

    total_loss = 0.8 * loss_front + 0.2 * loss_back
    return total_loss

# load input and output datasets
################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(9, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.w4 = nn.Conv3d(self.width, self.width, 1)
        self.w5 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.gelu(x)
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
################################################################
# configs
################################################################
mode1 =22
mode2=64
mode3=64
width = 12
batch_size = 16
batch_size2 = batch_size
epochs = 100
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5
train_size = 12000
test_size =1054
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(epochs,train_size,test_size,learning_rate, scheduler_step, scheduler_gamma)
################################################################
# training and evaluation
################################################################
r2_value=[]
train_loss=[]
test_loss=[]
best_r2=-float('inf')
best_model_state = None
model_save_path = '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/saved_models/'
model = FNO3d(mode1, mode2, mode3, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x).squeeze()
        #rint("输出",out.shape)
        #rint("y",y.shape)
        mse = F.mse_loss(out, y, reduction='mean')
        r2=r2_score(y.clone().detach(),out.clone().detach())
        l2 = compute_loss(out, y, myloss, batch_size)
        l2.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()
    r2_value.append(r2)
    if r2>best_r2:
        best_r2=r2
        best_model_state=model.state_dict()
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).squeeze()
            test_l2 += compute_loss(out, y, myloss, batch_size).item()
    train_mse /= len(train_loader)
    train_l2 /= train_size
    test_l2 /= test_size
    train_loss.append(train_l2)
    test_loss.append(test_l2)
    t2 = default_timer()
    print(ep,t2-t1,train_l2, test_l2,r2)
    if (ep+1) % 10== 0:
        save_file_path = os.path.join(model_save_path, f'co_all_layers_model_epoch_{ep+1}.pth')
        torch.save(model.state_dict(), save_file_path)
        print(f'Model saved at epoch {ep+1} to {save_file_path}')
max_r2=max(r2_value)
print("r2最大值",max_r2)
if best_model_state is not None:
    torch.save(best_model_state, '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/models/co_all_layers_best_model_1step.pth')
    print(f'最佳模型已保存，R²: {best_r2}')
epoch1=list(range(1,epochs+1))
plt.figure(1)
plt.plot(epoch1, train_loss, label='Train Loss')
plt.plot(epoch1, test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss and Test Loss')
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()
plt.figure(2)
plt.plot(epoch1, r2_value)
plt.xlabel('Epoch')
plt.ylabel('R2')
plt.title('R2 Score')
plt.show()
