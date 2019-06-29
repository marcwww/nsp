# nkbqa

## Introduction

This project is aiming at the NLP task knowledge-base question answering (KB-QA). The main reference paper is:

```
@inproceedings{jp2017scanner,
  author = {Jianpeng Cheng and Siva Reddy and Vijay Saraswat and Mirella Lapata},
  booktitle = {ACL},
  title = {Learning Structured Natural Language Representations for Semantic Parsing},
  year = {2017},
}
```
The model is transition-based, and the detailed Chinese explaination of this paper can be found in the file "Learning Structured Natural Language Representations for Semantic Parsing Jianpeng Cheng 阅后总结".

The original paper only support FunQL logic expression (corresponding to the folder ./nsp), and this project makes extension on it, with Lambda DCS supported.

## Dependencies

- Python2.7
- Numpy
- DyNet
- jpype

## Guide for Running this code

### Path Tree：

- nkbqa
  - data：for the datasets
    - free917：used in nsp
    - geoquery：used in nsp_lambda
  - lib：for the external dependency packages, mainly the .jar file from Standford project SEMPRE
    - lDCS_convertor.jar：used to parse Lambda-DCS logic expressions into parsing trees
  - mdl：for trained model dumps
    - lambda
    - prolog
  - nsp：the reproduce of the orginal paper
    - main.py：the main function, with training and testing
    - config.py：for the hyperparameters
    - ...
  - nsp_lambda：the version with Lambda-DCS support
    - main.py
    - config.py
    - ...
  - res：for the logs
    - lambda
    - prolog
  - wasp-1.0：an evaluation script, and actually the small knowledge base of GEOQUERY is in it.

### Running

One can directly run the main.py with python to train and test. If wanting testing only, the `--train` param should be omiited.

Train and test for nsp：

```shell
cd nkbqa
cd nsp
python main.py --train True
```

Train and test for nsp_lambda：

```Shell
cd nkbqa
cd nsp_lambda
python main.py --train True
```

**NOTICE** please ensure that only English in the absolute path str, or else the error `UnicodeDecodeError` will be raised.

## Experimental Results

For nsp, the metric is accuracy. After 100 epoches, acc. will be 72%

For nsp_lambda, the metric is accuracy in terms of structure matching, since without grounding. After 200 epoches, this acc. will be 57%.
