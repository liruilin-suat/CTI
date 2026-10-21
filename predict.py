# coding:utf8
import os
from util.config import ModelConf
from HyperCTI_torch import HyperCTI
import numpy as np
import torch
import pandas as pd
from time import strftime, localtime, time
import torch

from util.config import OptionConf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    dataset_name = "TCMsuite"

    conf = ModelConf(f'./HyperCTI_{dataset_name.lower()}.conf')
    if not conf.contains('datapath'):
        raise ValueError("配置文件中缺少 datapath")

    auc_list = []

    model_dir = f"saved_model/{dataset_name}"
    
    for _, fold in enumerate(os.listdir(model_dir)):
        if fold == "cv": continue
        i = int(fold.split("[")[-1].split("]")[0]) - 1
        print(f"\n🔁 处理第 {i} 折...")
        test_path = f'./datasets/{dataset_name}/train_datasets/test_fold_{i}.txt'
        model_ckpt_path = os.path.join(fold, "HyperCTI_model.pt")

        if not os.path.exists(test_path):
            raise ValueError(f"❌ 找不到测试文件: {test_path}")
        if not os.path.exists(os.path.join(model_dir, model_ckpt_path)):
            raise ValueError(f"❌ 找不到模型权重文件: {model_ckpt_path}")
            
        # 读取测试集
        test_set = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                herb_name, protein_name, label = parts
                test_set.append((herb_name, protein_name, int(float(label))))
        print(f"✅ 成功加载测试集，共 {len(test_set)} 条")

        model = HyperCTI(conf, train_set=[], test_set=test_set)
        model.initModel()
        # 加载保存的权重字典
        state_dict = torch.load(os.path.join(model_dir, model_ckpt_path))
        # 将权重加载到模型中
        model.model.load_state_dict(state_dict)
        # 切换到评估/推理模式（关闭 Dropout 和 BatchNorm 的更新）
        model.model.eval()
        predict = []
        label = []
        with torch.no_grad():
            for n, batch in enumerate(model.next_batch(data_type="test")):
                u_idx, i_idx, j_idx = batch
                # 2. 转换为 GPU Tensor
                u_idx_t = torch.from_numpy(np.array(u_idx, dtype=np.int64)).to(model.device)
                i_idx_t = torch.from_numpy(np.array(i_idx, dtype=np.int64)).to(model.device)
                j_idx_t = torch.from_numpy(np.array(j_idx, dtype=np.int64)).to(model.device)
                
                predict_logits = model.model(
                    u_idx_t, i_idx_t,
                    model.H_e, model.H_n, model.P_e, model.P_n,
                    model.pr_compound_emb, model.pr_protein_emb,
                    model.esm2_t, model.fp_t, model.smiles_t, model.gfeat_t, model.gadj_t
                )

                predict.extend(torch.sigmoid(predict_logits).cpu().numpy().tolist())
                label.extend(torch.clamp(j_idx_t, 0.0, 1.0).cpu().numpy().tolist())

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        res = pd.DataFrame({"label": label, "predict": predict})
        pred_res_save_dir = os.path.join(OptionConf(model.config["output.setup"])["-dir"], "cv")
        os.makedirs(pred_res_save_dir, exist_ok=True)
        res.to_csv(pred_res_save_dir + "/" + currentTime + "[" + str(i) + "]" + '.txt',
                          columns=['label', 'predict'], index=False)
        
        
