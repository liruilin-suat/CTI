from util.config import OptionConf
from util.dataSplit import *
from multiprocessing import Process, Manager
from util.io import FileIO
from time import strftime, localtime, time
import numpy as np
import os
from HyperCTI_torch import HyperCTI

def save_train_dataset(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir +"/one_zero.txt", "w", encoding="utf-8") as file:
        for row in data:
            line = "\t".join(map(str, row))
            file.write(line + "\n")


class HDR:
    def __init__(self, config):
        self.trainingData = []
        self.testData = []
        self.measure = []
        self.config = config
        self.ratingConfig = OptionConf(config['ratings.setup'])
        if self.config.contains('evaluation.setup'):
            self.evaluation = OptionConf(config['evaluation.setup'])
            if os.path.exists(config['dataset_dir'] + "/train_datasets/one_zero.txt"):
                self.trainingData = pd.read_csv(config['dataset_dir'] + "/train_datasets/one_zero.txt", 
                                                sep="\t", header=None, dtype={0: str, 1: str}).values.tolist()
            else:
                # self.trainingData = FileIO.loadDataSet(config, config['datapath'])
                self.trainingData = FileIO.loadDataSet(config, config['datapath'])
                save_train_dataset(self.trainingData, os.path.dirname(config['datapath']) + "/train_datasets")
        else:
            print('Wrong configuration of evaluation!')
            exit(-1)
        print('Reading data and preprocessing...')

    def execute(self):
        if self.evaluation.contains('-cv'):
            k = int(self.evaluation['-cv'])
            self.measure = []
            i = 1
            for train, test in DataSplit.crossValidation(
                self.trainingData, k, path=os.path.dirname(self.config['datapath']) + "/train_datasets"
            ):
                fold = '[' + str(i) + ']'
                print(f"=== Starting Fold {i}/{k} ===")
                
                model_instance = HyperCTI(self.config, train, test, fold=fold)
                fold_res = model_instance.trainModel()  # 直接拿到当前折的指标字典
                
                self.measure.append(fold_res)
                i += 1
            # -----------------------------------------------------------

            if not self.measure:
                print('所有折均未成功完成，请检查错误日志')
                return

            completed_k = len(self.measure)
            if completed_k < k:
                print(f'警告：{k} 折中只有 {completed_k} 折成功完成')

            # 获取所有指标的名称列表 ['AUC', 'Precision', 'Recall', 'F1']
            metric_names = self.measure[0].keys()

            # 计算各指标在所有折上的平均值，存入字典
            avg_res = {
                metric: sum(fold[metric] for fold in self.measure) / completed_k 
                for metric in metric_names
            }

            currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
            outDir = OptionConf(self.config['output.setup'])['-dir']
            fileName = self.config['model.name'] + '@' + currentTime + '-' + str(k) + '-fold-cv' + '.txt'
            FileIO.writeFile(outDir, fileName, avg_res)
            print('The result of %d-fold cross validation:\n%s' % (completed_k, avg_res))
        else:
            recommender = self.config['model.name'] + '(self.config,self.trainingData,self.testData)'
            eval(recommender).execute()

def run(measure, algor, order):
    measure[order] = algor.execute()