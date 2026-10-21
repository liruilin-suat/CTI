# coding: utf-8
import os
import time
from time import strftime, localtime, time
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import abspath
import pandas as pd
from random import shuffle

from util.config import OptionConf
from rating import Rating
import pandas as pd
from util.log import Log


class CrossAttentionGatedFusion(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.1):
        super(CrossAttentionGatedFusion, self).__init__()
        self.emb_size = emb_size
        
        # Multi-Head Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        # Modal Gating Network
        self.gate_fc = nn.Sequential(
            nn.Linear(emb_size * 3, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 3),
            nn.Softmax(dim=-1)
        )
        
        # MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, emb_size)
        )
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, h_smiles, h_fp, h_graph):
        """
        Input shapes: [batch_size, emb_size]
        """
        # Stack shape: [batch_size, 3, emb_size]
        Z = torch.stack([h_smiles, h_fp, h_graph], dim=1)
        
        # Cross-Modal Attention
        attn_out, _ = self.cross_attn(Z, Z, Z) # [batch_size, 3, emb_size]
        Z_enhanced = Z + attn_out              # residual connection
        
        # Gated Fusion
        concat_feats = Z_enhanced.reshape(Z_enhanced.size(0), -1) # [batch_size, 3 * emb_size]
        gate_weights = self.gate_fc(concat_feats).unsqueeze(-1)   # [batch_size, 3, 1]
        
        # sum -> [batch_size, emb_size]
        fused_vec = torch.sum(Z_enhanced * gate_weights, dim=1)
        
        # norm
        output = self.layer_norm(fused_vec + self.fusion_mlp(fused_vec))
        
        return output

class BioEncoder(nn.Module):
    def __init__(self, emb_size, esm2_dim, fp_dim, vocab_size, atom_feat_dim):
        super(BioEncoder, self).__init__()
        self.emb_size = emb_size
        cnn_half_dim = emb_size // 2

        # (a-1) ESM2 project
        self.fc1_esm2 = nn.Linear(esm2_dim, 512)
        self.fc2_esm2 = nn.Linear(512, emb_size)

        # (a-2) SMILES fp project
        self.fc_fp1 = nn.Linear(fp_dim, 512)
        self.fc_fp2 = nn.Linear(512, emb_size)
        
        # (a-3) SMILES CNN
        smiles_kernels = [4, 6, 8] 
        self.smiles_max_pool = nn.MaxPool1d(250-smiles_kernels[0]-smiles_kernels[1]-smiles_kernels[2]+3)
        self.smiles_embed = nn.Embedding(vocab_size, cnn_half_dim, padding_idx=0)
        self.smiles_cnn = nn.Sequential(
            nn.Conv1d(in_channels=cnn_half_dim, out_channels= cnn_half_dim,  kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_half_dim, out_channels= cnn_half_dim*2,  kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_half_dim*2, out_channels=cnn_half_dim*4,  kernel_size=6),
            nn.ReLU(),
        )
        self.fc_smiles = nn.Linear(cnn_half_dim*4, emb_size)
        
        # (a-4) GCN
        self.fc_gcn1 = nn.Linear(atom_feat_dim, 128)
        self.fc_gcn2 = nn.Linear(128, emb_size)

        # (a-5) Attention-based Fusion
        self.ln_seq = nn.LayerNorm(emb_size)
        self.ln_fp = nn.LayerNorm(emb_size)
        self.ln_gph = nn.LayerNorm(emb_size)
        
        # ablation w/o fusion
        # self.fusion_head = nn.Linear(emb_size, emb_size)
        # self.fusion_mlp = nn.Sequential(
        #     nn.Linear(emb_size, emb_size * 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(emb_size * 2, emb_size)
        # )
    
        self.fusion_module = CrossAttentionGatedFusion(emb_size=emb_size, num_heads=4)
        
        # norm
        self.ln_protein_out = nn.LayerNorm(emb_size)
        self.ln_fp_out = nn.LayerNorm(emb_size)
        self.ln_graph_out = nn.LayerNorm(emb_size)
        # ablation w/o fusion
        # self.ln_compound_out = nn.LayerNorm(emb_size)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, esm2_tensor, fp_tensor, smiles_tensor, gfeat_tensor, gadj_tensor):
        # (a-1) Protein
        h_protein_bio_1 = F.leaky_relu(self.fc1_esm2(esm2_tensor))
        h_protein_bio = self.ln_protein_out(F.leaky_relu(self.fc2_esm2(h_protein_bio_1)))

        # (a-2) Fingerprint
        h_fp = self.ln_fp_out(self.fc_fp2(F.relu(self.fc_fp1(fp_tensor))))
        
        # (a-3) smiles
        smiles_emb = self.smiles_embed(smiles_tensor)
        smiles_emb = smiles_emb.permute(0, 2, 1)
        smiles_cnn = self.smiles_cnn(smiles_emb)
        smiles_cnn = self.smiles_max_pool(smiles_cnn).squeeze(2)
        h_smiles = self.fc_smiles(smiles_cnn)
        
        # (a-4) Graph GCN
        h_gcn = F.relu(torch.matmul(gadj_tensor, self.fc_gcn1(gfeat_tensor)))
        h_gcn = F.relu(torch.matmul(gadj_tensor, self.fc_gcn2(h_gcn)))
        h_graph = self.ln_graph_out(h_gcn.mean(dim=1))

        # (a-5) Fusion Attention
        ln_seq = self.ln_seq(h_smiles)
        ln_fp = self.ln_fp(h_fp)
        ln_gph = self.ln_gph(h_graph)

        # Cross-Attention Gated Network
        h_compound_bio = self.fusion_module(ln_seq, ln_fp, ln_gph)

        # Z = torch.stack([ln_seq, ln_fp, ln_gph], dim=1)
        # Z = self.fusion_head(Z)
        # fused_vec = torch.sum(Z, dim=1)
        # fused_vec = self.fusion_mlp(fused_vec)
        # h_compound_bio = self.ln_compound_out(fused_vec)

        return h_compound_bio, h_protein_bio


class HyperCTI_Net(nn.Module):
    def __init__(self, num_compounds, num_proteins, emb_size, bio_config, n_layer=3):
        super(HyperCTI_Net, self).__init__()
        self.num_compounds = num_compounds
        self.num_proteins = num_proteins
        self.emb_size = emb_size
        self.n_layer = n_layer

        # bio feature encoder
        self.bio_encoder = BioEncoder(
            emb_size=emb_size,
            esm2_dim=bio_config['esm2_dim'],
            fp_dim=bio_config['fp_dim'],
            vocab_size=bio_config['vocab_size'],
            atom_feat_dim=bio_config['atom_feat_dim']
        )

        # hypergraph Node Embedding
        self.compound_embeddings = nn.Parameter(torch.FloatTensor(num_compounds, emb_size))
        self.protein_embeddings = nn.Parameter(torch.FloatTensor(num_proteins, emb_size))

        # hypergraph relationship gate network 
        self.gating1 = nn.Linear(emb_size, emb_size)
        self.gating2 = nn.Linear(emb_size, emb_size)

        # hypergraph learning 
        self.weights_layer = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=False) for _ in range(n_layer)])
        self.weights_layer_1 = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=False) for _ in range(n_layer)])

        # norm
        self.h_compound_norm = nn.LayerNorm(emb_size)
        self.h_protein_norm = nn.LayerNorm(emb_size)
        
        concat_dim = (emb_size + emb_size) * 2
        # concat_dim = (emb_size + emb_size)

        self.dropout1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(concat_dim, 512)
        
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(512, 256)
        
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(256, 1)
        
        self.norm_before_head = nn.LayerNorm(256)
        self.leaky_relu = nn.LeakyReLU()
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def self_gating(self, em, gating_layer):
        return em * torch.sigmoid(gating_layer(em))

    def forward_embeddings(self, H_e, H_n, P_e, P_n, pr_compound_emb, pr_protein_emb,
                           esm2_tensor, fp_tensor, smiles_tensor, gfeat_tensor, gadj_tensor):
        
        h_compound_bio, h_protein_bio = self.bio_encoder(
            esm2_tensor, fp_tensor, smiles_tensor, gfeat_tensor, gadj_tensor
        )

        # ablation hyerCTI-g: remove self-gating network
        compound_emb = self.self_gating(self.compound_embeddings, self.gating1)
        protein_emb = self.self_gating(self.protein_embeddings, self.gating2)
        
        # compound_emb = self.self_gating(h_compound_bio, self.gating1)
        # protein_emb = self.self_gating(h_protein_bio, self.gating2)

        # compound_emb = self.compound_embeddings
        # protein_emb = self.protein_embeddings

        for i in range(self.n_layer):
            new_hc_edge = torch.sparse.mm(H_e, compound_emb)
            # ablation hyerCTI-p: remove Page-Rank Integration mechanism
            # new_compound_emb = torch.sparse.mm(H_n, new_hc_edge)
            new_compound_emb = torch.sparse.mm(H_n, new_hc_edge) * pr_compound_emb

            new_pd_edge = torch.sparse.mm(P_e, protein_emb)
            # ablation hyerCTI-p: remove Page-Rank Integration mechanism
            # new_protein_emb = torch.sparse.mm(P_n, new_pd_edge)
            new_protein_emb = torch.sparse.mm(P_n, new_pd_edge) * pr_protein_emb
            
            new_compound_emb = F.leaky_relu(
                self.weights_layer[i](new_compound_emb) + compound_emb)
            new_protein_emb = F.leaky_relu(
                self.weights_layer_1[i](new_protein_emb) + protein_emb)

            compound_emb = F.normalize(new_compound_emb, p=2, dim=1)
            protein_emb = F.normalize(new_protein_emb, p=2, dim=1)

        hyper_compound_emb = compound_emb
        hyper_protein_emb = protein_emb

        final_compound_emb = torch.cat([hyper_compound_emb, h_compound_bio], dim=-1)
        final_protein_emb = torch.cat([hyper_protein_emb, h_protein_bio], dim=-1)

        # final_compound_emb = h_compound_bio
        # final_protein_emb = h_protein_bio

        return final_compound_emb, final_protein_emb

    def forward(self, u_idx, i_idx, H_e, H_n, P_e, P_n, pr_compound_emb, pr_protein_emb,
                esm2_tensor, fp_tensor, smiles_tensor, gfeat_tensor, gadj_tensor):
        
        # compound-target pair
        final_uemb, final_iemb = self.forward_embeddings(
            H_e, H_n, P_e, P_n, pr_compound_emb, pr_protein_emb,
            esm2_tensor, fp_tensor, smiles_tensor, gfeat_tensor, gadj_tensor
        )

        # compound/target feature index
        u_embed = final_uemb[u_idx]
        v_embed = final_iemb[i_idx]

        # concat and predict 
        pair = torch.cat([u_embed, v_embed], dim=-1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2_norm = self.norm_before_head(fully2)
        fully2_norm = self.dropout3(fully2_norm)
        predict = self.fc3(fully2_norm).squeeze(-1)

        return predict

class HyperCTI:
    def __init__(self, conf, train_set=None, test_set=None, fold='[1]'):
        self.config = conf
        self.foldInfo = fold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.modelName = self.config['model.name']
        self.output = OptionConf(self.config['output.setup'])
        self.isOutput = self.output.isMainOn()
        self.emb_size = int(self.config['num.factors'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        learningRate = OptionConf(self.config['learnRate'])
        self.lRate = float(learningRate['-init'])
        self.maxLRate = float(learningRate['-max'])
        self.batch_size = int(self.config['batch_size'])
        regular = OptionConf(self.config['reg.lambda'])
        self.regU,self.regI,self.regB= float(regular['-u']),float(regular['-i']),float(regular['-b'])
        self.initializing_log()
        
        self.data = Rating(self.config, train_set, test_set)
        # self.data = Rating(self.config, train_set, test_set, int(fold.split("[")[-1].split("]")[0]) - 1)
        self.num_compounds, self.num_proteins = self.data.cpSize()
        self.num_herbs, self.num_diseases, self.train_size = self.data.trainingSize()
        
        
    def initializing_log(self):
        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.log = Log(self.modelName, self.modelName + self.foldInfo + ' ' + currentTime)
        # save configuration
        self.log.add('### model configuration ###')
        for k in self.config.config:
            self.log.add(k + '=' + self.config[k])
    
    def printAlgorConfig(self):
        "show model's configuration"
        print('Model:', self.config['model.name'])
        print('Ratings dataset:', abspath(self.config['datapath']))
        if OptionConf(self.config['evaluation.setup']).contains('-testSet'):
            print('Test set:', abspath(OptionConf(self.config['evaluation.setup'])['-testSet']))
        # print dataset statistics
        print('Training set size: (herb count: %d, disease count %d, record count: %d)' % (self.data.trainingSize()))
        print('Test set size: (herb count: %d, disease count %d, record count: %d)' % (self.data.testSize()))
        print('=' * 80)
        # print specific parameters if applicable
        if self.config.contains(self.config['model.name']):
            parStr = ''
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)
            print('=' * 80)
    
    def next_batch(self, data_type='train'):
        if data_type == 'train':
            data_list = self.data.trainingData
            shuffle(data_list) 
        elif data_type == 'val':
            data_list = self.data.valData
        elif data_type == 'test':
            data_list = self.data.testData
        else:
            raise ValueError("data_type must be 'train', 'val' or 'test'")

        data_size = len(data_list)

        for batch_id in range(0, data_size, self.batch_size):
            batch = data_list[batch_id : batch_id + self.batch_size]

            u_idx = [self.data.compound[item[0]] for item in batch]
            i_idx = [self.data.protein[item[1]] for item in batch]
            j_idx = [item[2] for item in batch]

            yield u_idx, i_idx, j_idx
    
    
    def _scipy_to_torch_sparse(self, coo_mat):
        indices = torch.from_numpy(np.vstack((coo_mat.row, coo_mat.col)).astype(np.int64))
        values = torch.from_numpy(coo_mat.data.astype(np.float32))
        shape = torch.Size(coo_mat.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def buildAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.trainingData:
            if int(pair[2]) != 0:
                row.append(self.data.compound[pair[0]])
                col.append(self.data.protein[pair[1]])
                entries.append(1)
        return sp.coo_matrix((entries, (row, col)), shape=(self.num_compounds, self.num_proteins), dtype=np.float32)

    def buildhcAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.hcassociation:
            row.append(self.data.herb[pair[0]])
            col.append(self.data.compound[pair[1]])
            entries.append(1)
        return sp.coo_matrix((entries, (row, col)), shape=(self.num_herbs, self.num_compounds), dtype=np.float32)

    def buildpdAdjacencyMatrix(self):
        row, col, entries = [], [], []
        for pair in self.data.pdassociation:
            row.append(self.data.protein[pair[0]])
            col.append(self.data.disease[pair[1]])
            entries.append(1)
        return sp.coo_matrix((entries, (row, col)), shape=(self.num_proteins, self.num_diseases), dtype=np.float32)

    def buildGraphAndPageRank(self, adjacency_matrix):
        G = nx.Graph()
        coo = adjacency_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            G.add_edge(i, j, weight=v)
        return nx.pagerank(G, alpha=0.85)

    def _load_bio_features(self):
        default_features_dir = f'{self.config["dataset_dir"]}/features/'
        default_compound_dir = f'{self.config["dataset_dir"]}/features/newfeatures/'
        default_protein_dir = f'{self.config["dataset_dir"]}/features/newfeatures/'
        compound_features_dir = os.environ.get('HyperCTI_COMPOUND_FEATURES_DIR', default_compound_dir)
        protein_features_dir = os.environ.get('HyperCTI_PROTEIN_FEATURES_DIR', default_protein_dir)

        id2compound = {v: k for k, v in self.data.compound.items()}
        id2protein = {v: k for k, v in self.data.protein.items()}

        def _load_id_list(path):
            with open(path, 'r', encoding='utf-8') as fh:
                return [ln.strip() for ln in fh if ln.strip()]

        def _build_reorder(id2name, feat_id_list, n_total):
            feat_map = {name: idx for idx, name in enumerate(feat_id_list)}
            return np.array([feat_map.get(id2name.get(k, '__unk__'), 0) for k in range(n_total)], dtype=np.int64)

        try:
            c_ids = _load_id_list(os.path.join(compound_features_dir, 'compound_ids.txt'))
            c_reorder = _build_reorder(id2compound, c_ids, self.num_compounds)
        except FileNotFoundError:
            c_reorder = np.arange(self.num_compounds, dtype=np.int64)

        try:
            p_ids = _load_id_list(os.path.join(protein_features_dir, 'target_ids.txt'))
            p_reorder = _build_reorder(id2protein, p_ids, self.num_proteins)
        except FileNotFoundError:
            p_reorder = np.arange(self.num_proteins, dtype=np.int64)

        # (protein_n, 1280)
        self.esm2_np = np.load(os.path.join(protein_features_dir, 'protein_esm2_emb.npy')).astype(np.float32)[p_reorder]
        # (compound_n, 250)
        self.smiles_np = np.load(os.path.join(compound_features_dir, 'c_smiles_tokens.npy')).astype(np.int64)[c_reorder]
        # (compound_n, 2513)
        self.fp_np = np.load(os.path.join(compound_features_dir, 'c_fingerprints.npy')).astype(np.float32)[c_reorder]
        # (compound_n, 120, 133)
        self.gfeat_np = np.load(os.path.join(compound_features_dir, 'c_graph_feat.npy')).astype(np.float32)[c_reorder]
        # (compound_n, 120, 120)
        self.gadj_np = np.load(os.path.join(compound_features_dir, 'c_graph_adj.npy')).astype(np.float32)[c_reorder]

    def initModel(self):
        self._load_bio_features()
        self.esm2_t = torch.from_numpy(self.esm2_np).to(self.device)
        self.fp_t = torch.from_numpy(self.fp_np).to(self.device)
        self.smiles_t = torch.from_numpy(self.smiles_np).to(self.device)
        self.gfeat_t = torch.from_numpy(self.gfeat_np).to(self.device)
        self.gadj_t = torch.from_numpy(self.gadj_np).to(self.device)

        # 构建规范超图结构矩阵
        hc = self.buildhcAdjacencyMatrix()
        pd_mat = self.buildpdAdjacencyMatrix()

        def _safe_inverse(x):
            x = np.asarray(x, dtype=np.float32)
            return np.divide(1.0, x, out=np.zeros_like(x), where=x != 0.0)

        H_c = hc.transpose()
        inv_D_hc_v = _safe_inverse(H_c.sum(axis=1).reshape(1, -1))
        inv_D_hc_e = _safe_inverse(H_c.sum(axis=0).reshape(1, -1))
        H_e_mat = (H_c.multiply(inv_D_hc_e)).transpose().tocoo()
        H_n_mat = (H_c.transpose().multiply(inv_D_hc_v)).transpose().tocoo()

        inv_D_P_v = _safe_inverse(pd_mat.sum(axis=1).reshape(1, -1))
        inv_D_P_e = _safe_inverse(pd_mat.sum(axis=0).reshape(1, -1))
        P_e_mat = (pd_mat.multiply(inv_D_P_e)).transpose().tocoo()
        P_n_mat = (pd_mat.transpose().multiply(inv_D_P_v)).transpose().tocoo()

        self.H_e = self._scipy_to_torch_sparse(H_e_mat)
        self.H_n = self._scipy_to_torch_sparse(H_n_mat)
        self.P_e = self._scipy_to_torch_sparse(P_e_mat)
        self.P_n = self._scipy_to_torch_sparse(P_n_mat)

        pr_compound = self.buildGraphAndPageRank(hc)
        pr_protein = self.buildGraphAndPageRank(pd_mat)
        self.pr_compound_emb = torch.from_numpy(
            np.array([pr_compound.get(i, 0) for i in range(self.num_compounds)], dtype=np.float32).reshape(-1, 1)
        ).to(self.device)
        self.pr_protein_emb = torch.from_numpy(
            np.array([pr_protein.get(i, 0) for i in range(self.num_proteins)], dtype=np.float32).reshape(-1, 1)
        ).to(self.device)

        bio_config = {
            'esm2_dim': self.esm2_np.shape[1],
            'fp_dim': self.fp_np.shape[1],
            'vocab_size': int(np.max(self.smiles_np)) + 1,
            # 'vocab_size': 54,  # SMILES TOKEN 
            'max_atoms': self.gfeat_np.shape[1],
            'atom_feat_dim': self.gfeat_np.shape[2]
        }

        # 实例化网络主体
        self.model = HyperCTI_Net(
            self.num_compounds,
            self.num_proteins,
            self.emb_size,
            bio_config,
            n_layer=int(self.config["n_layer"])
        ).to(self.device)

    def trainModel(self):
        self.initModel()

        # 定义优化器（此时 model.parameters() 自动包含 BioEncoder 中的全部权重）
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.lRate))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.lRate), weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.maxEpoch)

        print("[Training] 开始 PyTorch 端到端模型训练 (BioEncoder 参与反向传播)...")

        for epoch in range(self.maxEpoch):
            self.model.train()
            for n, batch in enumerate(self.next_batch(data_type="train")):
                u_idx, i_idx, j_idx = batch
                u_idx_t = torch.from_numpy(np.array(u_idx, dtype=np.int64)).to(self.device)
                i_idx_t = torch.from_numpy(np.array(i_idx, dtype=np.int64)).to(self.device)
                j_idx_t = torch.from_numpy(np.array(j_idx, dtype=np.float32)).to(self.device)

                optimizer.zero_grad()

                predict_logits = self.model(
                    u_idx_t, i_idx_t,
                    self.H_e, self.H_n, self.P_e, self.P_n,
                    self.pr_compound_emb, self.pr_protein_emb,
                    self.esm2_t, self.fp_t, self.smiles_t, self.gfeat_t, self.gadj_t
                )

                labels = torch.clamp(j_idx_t, 0.0, 1.0)

                # loss
                data_loss = F.binary_cross_entropy_with_logits(predict_logits, labels, reduction='sum')

                loss = data_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                print(f'training: {epoch + 1} batch {n} loss: {loss.item():.4f}')
            scheduler.step()

        # test and save
        self.model.eval()
        predict = []
        label = []
        with torch.no_grad():
            for n, batch in enumerate(self.next_batch(data_type="test")):
                u_idx, i_idx, j_idx = batch
                # 2. 转换为 GPU Tensor
                u_idx_t = torch.from_numpy(np.array(u_idx, dtype=np.int64)).to(self.device)
                i_idx_t = torch.from_numpy(np.array(i_idx, dtype=np.int64)).to(self.device)
                j_idx_t = torch.from_numpy(np.array(j_idx, dtype=np.int64)).to(self.device)
                
                predict_logits = self.model(
                    u_idx_t, i_idx_t,
                    self.H_e, self.H_n, self.P_e, self.P_n,
                    self.pr_compound_emb, self.pr_protein_emb,
                    self.esm2_t, self.fp_t, self.smiles_t, self.gfeat_t, self.gadj_t
                )

                predict.extend(torch.sigmoid(predict_logits).cpu().numpy().tolist())
                label.extend(torch.clamp(j_idx_t, 0.0, 1.0).cpu().numpy().tolist())

        currentTime = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        res = pd.DataFrame({"label": label, "predict": predict})
        pred_res_save_dir = os.path.join(OptionConf(self.config["output.setup"])["-dir"], "cv")
        os.makedirs(pred_res_save_dir, exist_ok=True)
        res.to_csv(pred_res_save_dir + "/" + currentTime + self.foldInfo + '.txt',
                          columns=['label', 'predict'], index=False)
        
        ckp_save_dir = self.config['checkpoint']
        model_dir = os.path.join(ckp_save_dir, currentTime + self.foldInfo)
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'HyperCTI_model.pt')
        torch.save(self.model.state_dict(), save_path)
        print('模型权重已保存：', save_path)
        
        auc = roc_auc_score(label, predict)
        aupr = average_precision_score(label, predict)
        
        print("=" * 60)
        print(f"       fold: {self.foldInfo} 模型评估结果        ")
        print("=" * 60)
        print(f"AUC: {auc:.6f} | AUPR: {aupr:.6f}")
        print("=" * 30)

        fold_result = {
            "AUC": auc,
            "AUPR": aupr,
        }
        return fold_result
    