import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 导入字体管理模块
from torch.utils.data import Dataset, DataLoader
import gzip
import numpy as np
import shutil

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 1. 增强的数据集下载函数
# =====================================================================
class CustomMNIST(Dataset):
    """自定义MNIST数据集类，包含更健壮的下载和加载机制"""
    
    # 备用镜像URL
    RESOURCES = [
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4188051af3d5e6f500435b918'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'bfd4c38d0a60200e93c8bded2a94ff71')
    ]
    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.transform = transform
        self.train = train
        
        if download:
            self.download()
            
        self.data, self.targets = self.load_data()
        
    def download(self):
        """如果本地不存在则下载数据集"""
        if os.path.exists(os.path.join(self.root, 'processed', 'training.pt')):
            print("MNIST数据集已存在，跳过下载")
            return
            
        print("下载MNIST数据集...")
        os.makedirs(self.root, exist_ok=True)
        
        for url, md5 in self.RESOURCES:
            filename = url.rpartition('/')[2]
            dest_path = os.path.join(self.root, filename)
            
            # 下载文件
            if not os.path.exists(dest_path):
                print(f"正在下载: {url}")
                torch.hub.download_url_to_file(url, dest_path)
                
            # 解压文件
            extracted_path = dest_path[:-3]  # 去掉.gz扩展名
            if not os.path.exists(extracted_path):
                print(f"正在解压: {filename}")
                with gzip.open(dest_path, 'rb') as f_in:
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
            print(f"已下载并解压: {filename}")
            
    def load_data(self):
        """从本地文件加载数据"""
        if self.train:
            images_path = os.path.join(self.root, 'train-images-idx3-ubyte')
            labels_path = os.path.join(self.root, 'train-labels-idx1-ubyte')
        else:
            images_path = os.path.join(self.root, 't10k-images-idx3-ubyte')
            labels_path = os.path.join(self.root, 't10k-labels-idx1-ubyte')
            
        with open(images_path, 'rb') as f:
            # 跳过前16字节的头部信息
            images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            
        with open(labels_path, 'rb') as f:
            # 跳过前8字节的头部信息
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
            
        return torch.from_numpy(images).float(), torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        
        # 应用转换
        if self.transform:
            img = self.transform(img)
            
        return img, target

# 2. 图像预处理和数据加载
# =====================================================================
# 定义图像预处理管道
transform = transforms.Compose([
    transforms.ToPILImage(),         # 将numpy数组转换为PIL图像
    transforms.ToTensor(),            # 将图像转换为PyTorch张量 (0-255的像素值变为0-1)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
])

# 创建数据目录（当前目录下）
data_dir = './mnist_data'
os.makedirs(data_dir, exist_ok=True)

# 加载训练数据
train_data = CustomMNIST(
    root=data_dir,
    train=True,
    transform=transform,
    download=True
)

# 加载测试数据
test_data = CustomMNIST(
    root=data_dir,
    train=False,
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=1000,
    shuffle=False
)

print("成功加载数据集:")
print(f"训练样本数: {len(train_data)}")
print(f"测试样本数: {len(test_data)}")

# 查看一个样本
sample, label = train_data[0]
print(f"样本形状: {sample.shape}, 标签: {label}")

# 3. 神经网络模型定义
# =====================================================================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义全连接层 (Linear)
        # 28 * 28=784像素输入 -> 512个神经元 -> 256个神经元 -> 10个输出 (对应0-9数字)
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # 添加dropout层防止过拟合 (按25%概率随机丢弃神经元)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 将图像数据展平为一维向量 (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 28 * 28)
        
        # 第一层：全连接 + ReLU激活函数 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二层：全连接 + ReLU激活函数
        x = F.relu(self.fc2(x))
        
        # 输出层：全连接 + LogSoftmax (更适合与NLLLoss配合)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# 4. 模型训练设置
# =====================================================================
# 选择设备 (优先使用GPU加速)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")
if str(device) == "cpu":
    print("提示：使用GPU可以大大加快训练速度。如果你有NVIDIA显卡，请确保安装了CUDA驱动程序。")

# 实例化模型并移到设备上
model = Net().to(device)

# 打印模型摘要
print("\n模型架构:")
print(model)

# 定义优化器 (Adam优化器，学习率0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学习率调度器 (每10个epoch学习率乘以0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 5. 训练循环
# =====================================================================
epochs = 10
train_losses = []  # 记录每个epoch的平均损失
train_accuracies = []  # 记录每个epoch的训练准确率
test_losses = []  # 记录每个epoch的测试损失
test_accuracies = []  # 记录每个epoch的测试准确率

for epoch in range(1, epochs + 1):
    model.train()  # 设置为训练模式 (启用dropout)
    train_loss = 0
    correct = 0
    total = 0
    
    # 训练一个epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # 移动数据到设备
        data, target = data.to(device), target.to(device)
        
        # 梯度归零 (重要！否则梯度会累积)
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失 (使用负对数似然损失)
        loss = F.nll_loss(output, target)
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        
        # 记录统计信息
        train_loss += loss.item() * data.size(0)  # 乘以batch size以计算总损失
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress = 100. * batch_idx / len(train_loader)
            print(f'Epoch: {epoch}/{epochs} [{batch_idx}/{len(train_loader)}]'
                  f' ({progress:.0f}%) | '
                  f'Loss: {loss.item():.4f} | LR: {current_lr:.6f}')
    
    # 更新学习率 (每个epoch结束后调用)
    scheduler.step()
    
    # 计算训练集的平均损失和准确率
    train_loss /= len(train_loader.dataset)  # 平均损失
    train_accuracy = 100. * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    # 在测试集上评估模型
    model.eval()  # 设置为评估模式 (禁用dropout)
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum将损失累加
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # 计算测试集的平均损失和准确率
    test_loss /= total  # 平均损失
    test_accuracy = 100. * correct / total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # 打印每个epoch的最终结果
    print(f'\nEpoch {epoch} 完成: ')
    print(f'训练损失: {train_loss:.6f}, 训练准确率: {train_accuracy:.2f}% | '
          f'测试损失: {test_loss:.6f}, 测试准确率: {test_accuracy:.2f}%\n')

# 保存最终模型
os.makedirs('./models', exist_ok=True)
model_path = './models/first_ai_model.pth'
torch.save(model.state_dict(), model_path)
print(f"模型已保存为 '{model_path}'")

# 6. 结果可视化
# =====================================================================
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'o-', label='训练损失')
plt.plot(test_losses, 'o-', label='测试损失')
plt.title('训练和测试损失')
plt.xlabel('Epoch')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'o-', label='训练准确率')
plt.plot(test_accuracies, 'o-', label='测试准确率')
plt.title('训练和测试准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = './training_results.png'
plt.savefig(plot_path)
print(f"训练结果图表已保存为 '{plot_path}'")
plt.show()

# 查看学习率变化
final_lr = optimizer.param_groups[0]['lr']
print(f'\n最终学习率: {final_lr:.8f}')

print("\n训练完成！您已经成功训练了您的第一个人工智能模型！🎉")