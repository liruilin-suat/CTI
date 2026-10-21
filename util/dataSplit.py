from random import sample
from io import FileIO
import os
import pandas as pd

class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def crossValidation(data, k, output=True, path='./datasets/Symmap', val_ratio=0.1):
        if k <= 1 or k > 10:
            k = 3

        for i in range(k):
            if os.path.exists(os.path.join(path, f"test_fold_{i}.txt")) and os.path.exists(os.path.join(path, f"train_fold_{i}.txt")):
                trainingSet = pd.read_csv(
                    os.path.join(path, f"train_fold_{i}.txt"), sep="\t", header=None,  dtype={0: str, 1: str}
                    ).values.tolist()
                testSet = pd.read_csv(
                    os.path.join(path, f"test_fold_{i}.txt"),  sep="\t", header=None,  dtype={0: str, 1: str}
                    ).values.tolist()
            else:
                trainingSet = []
                testSet = []

                for ind, line in enumerate(data):
                    if ind % k == i:
                        testSet.append(line[:])
                    else:
                        trainingSet.append(line[:])
                # 如果开启了保存功能
                if output:
                    os.makedirs(path, exist_ok=True)

                    # train datasets
                    save_path = f"{path}/train_fold_{i}.txt"
                    with open(save_path, 'w', encoding='utf-8') as f:
                        for item in trainingSet:
                            # 假设每一项是一个可转为字符串的列表
                            if isinstance(item, list):
                                f.write('\t'.join(map(str, item)) + '\n')
                            else:
                                f.write(str(item) + '\n')
                    
                    # test datasets
                    save_path = f"{path}/test_fold_{i}.txt"
                    with open(save_path, 'w', encoding='utf-8') as f:
                        for item in testSet:
                            if isinstance(item, list):
                                f.write('\t'.join(map(str, item)) + '\n')
                            else:
                                f.write(str(item) + '\n')

            yield trainingSet, testSet
