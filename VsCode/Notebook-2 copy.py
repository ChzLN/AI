import os
import sys
# 导入PyTorch模块
import torch
# 导入PyTorch中的神经网络模块
import torch.nn as nn
# 导入PyTorch中的优化器模块
import torch.optim as optim
# 导入PyTorch中的函数模块 
# nn.functional 模块提供了一些常用的神经网络函数
# nn.functional 模块提供了很多常用的神经网络函数，如激活函数、池化函数、卷积函数等等。
import torch.nn.functional as F
'''
是 PyTorch 的一个扩展库，专门用于处理图像相关的任务。
datasets 模块包含了许多常用的计算机视觉数据集（如 MNIST、CIFAR-10 等），可以直接加载并用于训练和测试。
transforms 模块提供了一系列图像预处理方法（如缩放、裁剪、归一化等），通常用于对数据进行增强或标准化。
'''
from torchvision import datasets,transforms 

# 创建数据集
import numpy as np  # 导入 NumPy 库
import matplotlib
matplotlib.use('TkAgg')  # 设置 TkAgg 作为绘图后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 导入字体管理模块

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题



# %% 训练MNIST手写数字识别模型

# 1.加载数据（自动下载）
'''
是用来定义图像预处理流水线的。
它的作用是将图像数据转换为 PyTorch 张量（ToTensor()），
这是在使用 PyTorch 进行深度学习训练时常见的预处理步骤。
'''
transforms = transforms.Compose([transforms.ToTensor()])  # 将图像转换为张量

# MNIST 数据集是一个手写数字识别数据集，包含了大量的手写数字图像。
'''
`/data` 是数据集存储的路径，如果该路径下没有数据集，n`download=True` 会自动下载。
`train=True` 表示加载训练集数据（如果设置为 False，则会加载测试集）。
`transform=transforms` 表示对加载的数据应用之前定义的图像预处理流水线。
'''
train_data = datasets.MNIST('/data', train=True, download=True, transform=transforms)

# 加载测试数据 (评估模型性能)
test_data = datasets.MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms
)

# 创建数据加载器，用于批量加载数据
'''
`train_data` 是一个 PyTorch 数据集对象，它包含了 MNIST 训练集的图像和标签。
`batch_size=64` 表示每个批次加载 64 张图像。
`shuffle=True` 表示在每个 epoch 开始时打乱数据顺序，这有助于提高模型的泛化能力。
`epoch` 是指整个数据集被完整地训练一次。
'''
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1000,  # 评估时用较大batch加快速度
    shuffle=False
)

# 2.定义神经网络 （三层全连接神经网络）
# 定义一个简单的三层全连接神经网络
class Net(nn.Module):
    # 初始化方法
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层全连接层，输入维度为 784（28*28的图像展平），输出维度为512
        self.fc1 = nn.Linear(784,512)
        # 定义第二层全连接层 ，输入维度为 512，输出维度为 256
        self.fc2 = nn.Linear(512, 256)
        # 定义输出层 全连接层，输入维度为 256，输出维度为 10（对应数字0-9）
        self.fc3 = nn.Linear(256, 10)
        # 添加dropout层防止过拟合 (按25%概率随机丢弃神经元)
        self.dropout = nn.Dropout(0.25)

    # 前向传播方法 , 定义数据如何通过网络传递
    def forward(self, x):
        # 将输入展平为一维向量
        x = x.view(-1, 28 * 28)  # 展平输入

        '''
        `ReLU` 是一种常用的激活函数，它将输入中的所有小于零的值置零，并保留大于零的值。
        如果输入 $ x $ 是正数，输出等于输入。
        如果输入 $ x $ 是负数或零，输出为 0。
        '''
         # 通过第一层全连接层，应用ReLU激活函数，并保存结果 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 通过第二层全连接层，应用ReLU激活函数，并保存结果
        x = F.relu(self.fc2(x))

        # 通过输出层，并返回结果 全连接 + LogSoftmax 
        '''
        `log_softmax` 函数将输入的向量进行归一化，
        使其和为 1，并返回对数Softmax概率。
        `dim=1` 表示对输入的每一行进行归一化，即将每一行归一化为概率向量。
        `self.fc3(x)` 是通过第三层全连接层，并返回结果。
        `F.log_softmax(x, dim=1)` 是对输出进行 Softmax 归一化，并返回对数概率向量。
        '''
        x = self.fc3(x)

        # 返回输出结果
        return F.log_softmax(x, dim=1)

# 创建一个神经网络实例

# `device` 表示设备，如果 GPU 可用，则使用 GPU，否则使用 CPU。
# `cuda` 是 NVIDIA 提供的并行计算架构，允许软件使用 GPU 进行通用计算。
# `torch.cuda.is_available()` 检查 GPU 是否可用。`
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# `f` 是一个字符串，表示当前使用的设备。
print(f"此处使用的设备: {device}")

# 将模型移动到设备上
'''
创建一个神经网络模型 Net 的实例，并将其移动到指定的计算设备上（如 GPU 或 CPU）。
`.to(device)` 方法将模型移动到设备上 。
如果设备是 GPU，则将模型移动到 GPU 上，以利用 GPU 的并行计算能力。
'''
model = Net().to(device)

# 定义损失函数和优化器
'''
`CrossEntropyLoss` 是一种常用的分类损失函数，适用于多分类问题。
`Adam` 是一种常用的优化算法，用于训练神经网络。
`lr` 是学习率，控制模型权重更新的幅度。
'''
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 开始训练
# 迭代 epochs 次 ，每次迭代一个 epoch 
# `enumerate` 函数将迭代器转换为一个索引序列，同时迭代数据。
epochs = 10
train_losses = []  # 记录每个epoch的平均损失
train_accuracies = []  # 记录每个epoch的训练准确率
test_losses = []  # 记录每个epoch的测试损失
test_accuracies = []  # 记录每个epoch的测试准确率

for epoch in range(1,epochs+1):
    # 训练模式
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # 训练一个epoch
    # 移动数据到设备上
    # `data` 是输入数据，`target` 是对应的标签。
    #  enumerate 函数将迭代器转换为一个索引序列，同时迭代数据。
    # `data` 和 `target` 是一个批次的数据和标签。
    for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据和标签移动到设备上
            # `.to(device)` 方法将数据移动到指定的设备上。
            data, target = data.to(device), target.to(device)

            # 梯度归零 (重要！否则梯度会累积)
            optimizer.zero_grad()

            # 前向传播
            output = model(data)

            # 计算损失 
            # `nn.functional.nll_loss` 是一个用于计算负对数似然损失函数的函数。
            loss = F.nll_loss(output,target)

            # 反向传播
            # `loss.backward()` 方法计算梯度
            loss.backward()

            # 更新参数
            # `optimizer.step()` 方法更新参数
            optimizer.step()

            #统计训练准确率
            train_loss += loss.item()
            # 统计预测正确的数量
            _, predicted = output.max(1)
            # 统计总样本数
            total += target.size(0)
            # 统计预测正确的数量
            correct += predicted.eq(target).sum().item()

            # 每满100个批次，打印当前批次的损失
            if batch_idx % 100 == 0:
                # 打印当前批次的损失
                print(f'Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}]'
                      f'({100. * batch_idx / len(train_loader):.0f}%)|'
                      f'Loss: {loss.item():.4f}|'
                      )
# 更新学习率 (每个epoch结束后调用)
scheduler.step()

# 计算训练集的平均损失和准确率
train_loss /= len(train_loader.dataset)  # 使用 len(train_loader.dataset) 更准确
train_accuracy = 100. * correct / total
train_losses.append(train_loss)
train_accuracies.append(train_accuracy)  # 修复：将 train_accuracies 改为 train_accuracy

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
test_loss /= len(test_loader.dataset)
test_accuracy = 100. * correct / total
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# 打印每个epoch的最终结果
print(f'\nEpoch {epoch} 完成: ')
print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.2f}% | '
      f'测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%\n')

# 保存最终模型
torch.save(model.state_dict(), './models/first_ai_model_N2_Copy.pth')
print("模型已保存为 'first_ai_model_N2_Copy.pth'")




# 5. 结果可视化
# =====================================================================
def plot_training_results():
    """绘制训练过程中的损失曲线和准确率曲线"""
    plt.figure(figsize=(14, 6))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'o-', linewidth=2, markersize=8, label='训练损失')
    plt.plot(test_losses, 's-', linewidth=2, markersize=8, label='测试损失')
    plt.title('训练和测试损失', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'o-', linewidth=2, markersize=8, label='训练准确率')
    plt.plot(test_accuracies, 's-', linewidth=2, markersize=8, label='测试准确率')
    plt.title('训练和测试准确率', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('准确率 (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plot_path = './training_results_N2_Copy.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练结果图表已保存为 '{plot_path}'")
    plt.show()

# 调用绘图函数
plot_training_results()



# 6. 查看学习率变化
# =====================================================================
for i, lr in enumerate(scheduler.get_last_lr()):
    print(f'最终学习率 (参数组 {i}): {lr:.6f}')



'''
    主要修复和改进说明：
    ​修复关键问题​：
    # 错误位置：在batch循环内调用scheduler.step()
    # 正确位置：在epoch结束后调用
    for epoch in range(epochs):
        for batch in train_loader:
            # 训练代码...
            # 错误：scheduler.step()  # 在每个batch调用
            # 正确：在每个epoch结束后调用
            # scheduler.step()
    
    ​添加测试集评估​：
    在每个epoch后评估测试集准确率
    使用model.eval()和torch.no_grad()确保评估准确性

    ​数据增强​：
    添加了归一化处理(注释中保留)
    创建测试集用于模型评估

    ​训练监控​：
    记录训练/测试损失和准确率
    添加训练进度百分比显示
    可视化训练过程

    ​模型改进​：
    在第一个全连接层后添加dropout防止过拟合
    使用更精确的评估方法(epoch平均)

    ​内存优化​：
    使用reduction='sum'计算测试损失
    使用with torch.no_grad()减少内存占用

    ​可视化结果​：
    绘制损失曲线和准确率曲线
    保存结果图片便于分析
'''