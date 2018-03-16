# nkbqa

[TOC]

姓名：王克欣 

学号：201728014628049

学院：人工智能学院

## 简介

本项目处理的任务是知识库问答。主要参考文献为

```
@inproceedings{jp2017scanner,
  author = {Jianpeng Cheng and Siva Reddy and Vijay Saraswat and Mirella Lapata},
  booktitle = {ACL},
  title = {Learning Structured Natural Language Representations for Semantic Parsing},
  year = {2017},
}
```

采用的主要方法是通过状态转移模型进行神经语义解析，主要原理可以参考附件“Learning Structured Natural Language Representations for Semantic Parsing Jianpeng Cheng 阅后总结“。

​	原有工作只支持$\mathsf{FunQL}$的目标逻辑形式，本项目在原有工作（本项目复现版本对应文件夹nsp）基础上进行了扩展，加入了支持$\mathsf{Lambda\ DCS}$的版本（对应文件夹nsp_lambda）的代码实现。

## 依赖环境

- Python2.7
- Numpy
- DyNet
- jpype

## 运行说明

### 文件目录：

- nkbqa
  - data：存放数据集
    - free917：针对nsp
    - geoquery：针对nsp_lambda
  - lib：项目依赖$\mathsf{jar}$包，来源于斯坦福项目sempre
    - lDCS_convertor.jar：本项目中用来对$\mathsf{Lambda\ DCS}$逻辑表达式进行解析获得语法树
  - mdl：存放训练好的模型
    - lambda
    - prolog
  - nsp：原有工作的复现版本
    - main.py：主函数，训练及测试
    - config.py：存放参数设置
    - ...
  - nsp_lambda：支持$\mathsf{Lambda\ DCS}$的版本
    - main.py：主函数，训练及测试
    - config.py
    - ...
  - res：存放运行结果
    - lambda
    - prolog
  - wasp-1.0：存放$\mathsf{prolog}$评测脚本，以及$\mathsf{geoquery}$数据集的知识库

### 运行

可直接使用python运行两个项目的main.py文件运行，依次进行训练和测试（如果不进行训练可以不加参数`--train`）。

训练并测试nsp：

```shell
cd nkbqa
cd nsp
python main.py --train True
```

训练并测试nsp_lambda：

```Shell
cd nkbqa
cd nsp_lambda
python main.py --train True
```

**注意:** 请放置与英文路径下运行，否则会报错`UnicodeDecodeError`。

## 实验结果

对于nsp，评测指标为准确率，经100轮训练（默认参数设置下），该指标可达到72%。

对于nsp_lambda，由于没有实现grouding操作，仅仅可以生成中间表达，这里仅评测逻辑表达式的结构准确率。经200轮训练（默认参数设置下），逻辑表达式结构准确率可达到57%。