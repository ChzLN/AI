import sys
# 导入PyTorch模块
import torch
# 导入PyTorch中的神经网络模块
import torch.nn as nn
# 导入PyTorch中的优化器模块
import torch.optim as optim
'''
是 PyTorch 的一个扩展库，专门用于处理图像相关的任务。
datasets 模块包含了许多常用的计算机视觉数据集（如 MNIST、CIFAR-10 等），可以直接加载并用于训练和测试。
transforms 模块提供了一系列图像预处理方法（如缩放、裁剪、归一化等），通常用于对数据进行增强或标准化。
'''
from torchvision import datasets,transforms 

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
`/data` 是数据集存储的路径，如果该路径下没有数据集，
`download=True` 会自动下载。
`train=True` 表示加载训练集数据（如果设置为 False，则会加载测试集）。
`transform=transforms` 表示对加载的数据应用之前定义的图像预处理流水线。
'''
train_data = datasets.MNIST('/data', train=True, download=True, transform=transforms)
# 创建数据加载器，用于批量加载数据
'''
`train_data` 是一个 PyTorch 数据集对象，它包含了 MNIST 训练集的图像和标签。
`batch_size=64` 表示每个批次加载 64 张图像。
`shuffle=True` 表示在每个 epoch 开始时打乱数据顺序，这有助于提高模型的泛化能力。
`epoch` 是指整个数据集被完整地训练一次。
'''
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

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

    # 前向传播方法 , 定义数据如何通过网络传递
    def forward(self, x):
        # 将输入展平为一维向量
        x = x.view(-1, 28 * 28)  # 展平输入
        '''
        `ReLU` 是一种常用的激活函数，它将输入中的所有小于零的值置零，并保留大于零的值。
        如果输入 $ x $ 是正数，输出等于输入。
        如果输入 $ x $ 是负数或零，输出为 0。
        '''
         # 通过第一层全连接层，应用ReLU激活函数，并保存结果
        x = torch.relu(self.fc1(x))
        # 通过第二层全连接层，应用ReLU激活函数，并保存结果
        x = torch.relu(self.fc2(x))
        # 通过输出层，并返回结果
        '''
        `log_softmax` 函数将输入的向量进行归一化，
        使其和为 1，并返回对数Softmax概率。
        `dim=1` 表示对输入的每一行进行归一化，即将每一行归一化为概率向量。
        `self.fc3(x)` 是通过第三层全连接层，并返回结果。
        '''
        return torch.log_softmax(self.fc3(x), dim=1)
    
    # 3.训练模型
    
    # 创建一个神经网络实例
    '''
    `device` 表示设备，如果 GPU 可用，则使用 GPU，否则使用 CPU。
    `cuda` 是 NVIDIA 提供的并行计算架构，允许软件使用 GPU 进行通用计算。
    `torch.cuda.is_available()` 检查 GPU 是否可用。`
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    for epoch in range(epochs):
        # 移动数据到设备上
        # `data` 是输入数据，`target` 是对应的标签。
        #  enumerate 函数将迭代器转换为一个索引序列，同时迭代数据。
        # `data` 和 `target` 是一个批次的数据和标签。
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据和标签移动到设备上
            # `.to(device)` 方法将数据移动到指定的设备上。
            data, target = data.to(device), target.to(device)
            # 前向传播
            optimizer.zero_grad()
            # 通过模型计算输出
            output = model(data)
            # 计算损失 
            # `nn.functional.nll_loss` 是一个用于计算负对数似然损失函数的函数。
            loss = nn.functional.nll_loss(output,target)
            # 反向传播
            # `loss.backward()` 方法计算梯度
            loss.backward()
            # 更新参数
            # `optimizer.step()` 方法更新参数
            optimizer.step()
            # `scheduler.step()` 方法更新学习率
            scheduler.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} Loss:{loss.item():.4f}')
    
    torch.save(model.state_dict(), 'first_ai_model.pth')
            