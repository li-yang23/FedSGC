# FedSGC: Federated Simple Graph Convolution for Node Classification

**Disclaimer:** This code is an independent implementation of the methods described in the conference paper [FedSGC: Federated Simple Graph Convolution for Node Classification](https://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_5.pdf) and is **NOT** the official version. Use at your own risk; the author assumes no responsibility for any discrepancies or damages resulting from its use.

SGC implementation is cloned from official implementation for the paper [Simplifying Graph Convolutional Networks](https://github.com/Tiiiger/SGC). I just add a new python script for FedSGC implementation.

Experiments(on Macbook Pro 2019) for now include:

|Dataset|Train Metrics|Test Metrics|Training time|
| :---: |    :---:    |    :---:   |    :---:    |
|  Cora |  Acc:79.00% | Acc:80.80% |    2.03s    |

**environment preparation** (on Python 3.11.9, not promised, at least work for me)

```bash
pip install requirements.txt
```

To use it:

```bash
python fedsgc.py
```

