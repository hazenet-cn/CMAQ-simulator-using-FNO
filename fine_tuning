import os
import torch
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, input_folder_path, output_folder_path,mete_folder_path):
        self.input_files = sorted([os.path.join(input_folder_path, f) for f in os.listdir(input_folder_path) if f.endswith('.pt')])
        self.output_files = sorted([os.path.join(output_folder_path, f) for f in os.listdir(output_folder_path) if f.endswith('.pt')])
        self.mete_files=sorted([os.path.join(mete_folder_path, f) for f in os.listdir(mete_folder_path) if f.endswith('.pt')])
        assert len(self.input_files) == len(self.output_files), "Input and output files count do not match!"        
    def __len__(self):
        return len(self.input_files)
    def __getitem__(self, idx):
        input_data = torch.load(self.input_files[idx])
        output_data = torch.load(self.output_files[idx])
        mete_data=torch.load(self.mete_files[idx])
        return input_data,output_data,mete_data,idx
input_folder_path = '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/co_input'
output_folder_path = '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/co_output'
mete_folder_path = '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/new_mete'
dataset = CustomDataset(input_folder_path, output_folder_path,mete_folder_path)
print(f'数据集中的数据数量: {len(dataset)}')
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=1)
for i, (input_data, output_data,mete_data,idx) in enumerate(dataloader):
    # 将数据移动到GPU（如果可用）
    if torch.cuda.is_available():
        input_data = input_data.to('cuda')
        output_data = output_data.to('cuda')
        mete_data=mete_data.to('cuda')
    print(f'处理第 {i+1} 个批次数据')
print('数据加载完成')
#######################################################
state_dict = torch.load( '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/models/co_all_layers_best_model_1step.pth')
model.load_state_dict(state_dict)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0

    for x, y, z, idx in train_loader:
        x, y, z = x.cuda(), y.cuda(), z.cuda()
        optimizer.zero_grad()
        total_l2 = 0
        out1 = model(x)
        out = out1.squeeze()
        y= y.squeeze()
        #print(out.shape,y.shape)
        r2=r2_score(y.clone().detach(),out.clone().detach())
        l2_1 = compute_loss(out, y, myloss, batch_size)
        total_l2 += l2_1  
        prev_output = out1
        prev_z = z
        for step in range(1, 12 + 1):
            next_idx = idx + step
            valid_idx = next_idx < len(dataset)
            if valid_idx.any():
                next_x, next_y, next_mete_data, _ = dataset[next_idx[valid_idx]]
                next_y = next_y.cuda()
                next_mete_data = next_mete_data.unsqueeze(0).cuda()
                x_input = torch.cat((prev_output, prev_z), dim=-1)
                prev_output = model(x_input)
                out = prev_output.squeeze()
                next_y=next_y.squeeze()
                l2 = compute_loss(out, next_y, myloss, batch_size)
                total_l2 += l2
                prev_z = next_mete_data
            else:
                break

        # Backward propagation and optimization
        total_l2.backward()
        optimizer.step()
        train_l2 += total_l2.item()
    r2_value.append(r2)
    if r2>best_r2:
        best_r2=r2
        best_model_state=model.state_dict()
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y,z,idx in test_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x).squeeze()
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    train_mse /= len(train_loader)
    train_l2 /= train_size
    test_l2 /= test_size
    train_loss.append(train_l2)
    test_loss.append(test_l2)
    t2 = default_timer()
    print(ep,t2-t1,train_l2, test_l2,r2)
    if (ep+1) % 1== 0:
        save_file_path = os.path.join(model_save_path, f'co_all_layers_12steps_tuned_model_epoch_{ep+1}.pth')
        torch.save(model.state_dict(), save_file_path)
        print(f'Model saved at epoch {ep+1} to {save_file_path}')
max_r2=max(r2_value)
print("r2最大值",max_r2)
if best_model_state is not None:
    torch.save(best_model_state, '/dssg/home/acct-esehazenet/hazenet-zhanglanyi/models/co_all_layers_12steps_tuned_best_model.pth')
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
