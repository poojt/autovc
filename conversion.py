import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:1'
G = Generator(32,256,512,32).eval().to(device)
# G = Generator(16,256,512,16).eval().to(device)

# g_checkpoint = torch.load('autovc.ckpt')
g_checkpoint = torch.load('autovc.ckpt')
G.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('metadata.pkl', "rb"))

spect_vc = []
# 遍历说话人信息，取源说话人信息 sbmt_i
for sbmt_i in metadata:

    # 取说话人说话梅尔数据
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    # 遍历说话人信息，取目标说话人信息 sbmt_j
    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            # 输入源说话人说话数据 uttr_org ，源说话人说话人编码 emb_org 与目标说话人说话人编码 emb_trg ，得到最终转换结果 x_identic_psnt
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            print('mel size:', x_identic_psnt.size())
            
        if len_pad == 0:
            # uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()
        else:
            # uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
            uttr_trg = x_identic_psnt[0, :-len_pad, :].cpu().numpy()
        # 将转换路径 sbmt_i[0], sbmt_j[0] 与转换后的语音数据 uttr_trg 合并，接着添加至转换信息与语音梅尔频谱列表 spect_vc
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)