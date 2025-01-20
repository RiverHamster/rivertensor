# 人工智能中的编程大作业

在这次的大作业里，希望你能把之前学过的知识运用到识别Mnist数据集中，做出一个简易的手写数字识别系统。请按照以下步骤来进行：

1. 我们已经在作业0中使用了 `PyTorch` 中的卷积网络完成`Mnist`数据集的图像分类, 可以将你已经实现的代码复制过来
2. 基于PyTorch，实现数据并行或模型并行
3. 利用自己实现的框架，完成Mnist数据集的图像分类
4. Bonus:用自己的框架实现ImageNet的图像分类

## Task 1: PyTorch Basic Implementation

### 目标

使用PyTorch中的卷积神经网络(CNN)来完成Mnist数据集的图像分类。

直接将你在hw0中的代码复制过来作为Task1即可

## Task 2: PyTorch Parallel Practice

### 目标

学习了解PyTorch的数据并行，模型并行以提高训练的效率。选择二者其一来完成。

### 步骤:

1. **并行准备:** 确认训练环境，检测当前可用的GPU数量。
2. **数据并行:** 将数据分给不同的GPU，进行同步更新，以并行计算。
3. **模型并行:** 将模型的不同部分分布在不同的GPU上进行并行计算。
4. **性能对比:** 对比并行化前后的训练速度和准确率。

### 参考资料

- [PyTorch Data Parallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [PyTorch Model Parallel](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
- [PyTorch Distributed and Parallel Tutorial Summary](https://pytorch.org/tutorials/distributed/home.html)
- [Dive into Deep Learning Multiple Gpus Tutorial](https://d2l.ai/chapter_computational-performance/multiple-gpus.html)

如果感兴趣可以阅读下面这个资料，介绍了含有实际训练中考虑到NV-Link等因素后的多GPU训练。

- [Multi-GPU Training in PyTorch with Code](https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-1-single-gpu-example-d682c15217a8)

### Tips

1. 可以用PyTorch的API也可以自己写逻辑进行实现。
2. 如果确实没有多GPU设备可以把数据或者模型都放于一个GPU或者CPU上，但是按照数据并行或模型并行的方式来进行同步，并对比并行化前后的训练速度和准确率（此过程可以不用PyTorch自带的API）。虽然用单卡模拟多卡并不会真正带来效率的提升，也请做前后的效率对比。

---

## Task 3: Custom Implementation

### 目标

不使用现成的深度学习框架，而是基于cuda、pybind11、python等语言来自主实现一个简单卷积网络的框架（不需要多卡），并完成Mnist数据集的图像分类任务。

### 步骤:

1. 使用Cuda完成卷积网络，包括卷积过程的正向过程和反向过程。
2. 使用pybind11来将Cuda的代码转换成Python可以调用的.pyd文件，最好设置成class。
3. 使用Python端写好自动微分以及优化器，然后定义合适的损失函数。来实现损失函数的优化，完成Mnist数据集的分类任务。

### Tips：

1. 请妥善利用数次小作业的结果来实现本次大作业。
2. 我们不对数据处理（dataloader制作和预处理）做任何格式的要求，因此你可以使用torchvision等库进行处理。
3. 我们只要求卷积网络必须使用Cuda实现。如果更熟悉Cpp编程，你也可以使用Cpp/Cuda来实现自动微分和优化器。

---

## 评分

评分标准旨在确保学生不仅追求算法的高性能，而且注重实施的质量和细节。以下是具体的评分细则：

### 1. 算法性能

#### 1.1. 准确率

#### 1.2. 训练速度

#### 1.3. 模型复杂度

### 2. 实施细节

#### 2.1. 代码质量

> 代码结构清晰，命名规范，注释充分

#### 2.2. 报告完整性

> 报告包含实验背景、方法、结果、讨论、结论

---

## 提交

- 大项目的提交截止日期是2024/1/26，也即期末周结束后的两周内。请于此连接提交三个板块的所有内容，请提交命名格式为`Project_学号_姓名.zip`的文件，有以下目录
- `Task1/`: 请提交Task1代码文件和对于板块使用文档和分析情况的`pdf`, 可以提交hw0的报告
- `Task2/`: 请提交Task2代码文件和对于板块使用文档和分析情况的`pdf`
- `Task3/`: 请提交Task3代码文件和对于板块使用文档和分析情况的`pdf`

---

## 注意事项

> - 请誉学术诚信，杜绝抄袭，一经发现，严肃处理。
> - 请于报告中写清楚如何运行你的代码，以及如何复现你的结果。
> - 请不要把test set的数据放入你的模型训练中。