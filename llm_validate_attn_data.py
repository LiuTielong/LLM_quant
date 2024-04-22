"""
现在Attn层的数据有些乱。而且有的线性层输出应该保存为int8, 有些应该保存为uint8.
我先把数据流打通, 然后查看一下各个层的数据范围, 矩阵大小, 决定数据形式。
"""
import torch
import numpy as np
import math
import struct
pth_prefix = "F:/layer0_4_16/model.decoder.layers.0."


def test_layernorm_in():
    QKV_in_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    QKV_in = torch.load(QKV_in_filename, map_location='cpu').detach().squeeze()
    
    zp_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/zeropoint.pth"
    zp = torch.load(zp_filename, map_location='cpu').detach().squeeze()

    print(f"Layernorm输入数据形状是: {QKV_in.shape}")
    print(f"Layernorm输入范围是:{torch.min(QKV_in)}, {torch.max(QKV_in)}")
    print(f"Layernorm输入零点形状是:{zp.shape}")
    print(f"Layernorm输入零点范围是:{torch.min(zp)}, {torch.max(zp)}")

    return

def test_layernorm_out():
    QKV_in_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/x_int_add_zp_clamp.pth"
    QKV_in = torch.load(QKV_in_filename, map_location='cpu').detach().squeeze()
    
    zp_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
    zp = torch.load(zp_filename, map_location='cpu').detach().squeeze()

    print(f"Layernorm输出数据形状是: {QKV_in.shape}")
    print(f"Layernorm输出范围是:{torch.min(QKV_in)}, {torch.max(QKV_in)}")
    print(f"Layernorm输出零点形状是:{zp.shape}")
    print(f"Layernorm输出零点范围是:{torch.min(zp)}, {torch.max(zp)}")
    return

def test_q_proj():
    W_filename = f"{pth_prefix}self_attn.q_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()

    w_zp_source_filename = f"{pth_prefix}self_attn.q_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()

    print(f"Q_proj的数值范围:{torch.min(W)}, {torch.max(w_zp)}")
    print(f"Q_proj的零点形状:{w_zp.shape}")
    print(f"Q_proj的零点范围:{torch.min(w_zp)}, {torch.max(w_zp)}")

    A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"Q_proj输出数据形状是: {A_out_true.shape}")
    print(f"Q_proj输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"Q_proj输出零点形状是:{A_out_zp.shape}")
    print(f"Q_proj输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")
    return

def test_k_proj():
    W_filename = f"{pth_prefix}self_attn.k_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()

    w_zp_source_filename = f"{pth_prefix}self_attn.k_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()

    print(f"k_proj的数值范围:{torch.min(W)}, {torch.max(w_zp)}")
    print(f"k_proj的零点形状:{w_zp.shape}")
    print(f"k_proj的零点范围:{torch.min(w_zp)}, {torch.max(w_zp)}")

    A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"k_proj输出数据形状是: {A_out_true.shape}")
    print(f"k_proj输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"k_proj输出零点形状是:{A_out_zp.shape}")
    print(f"k_proj输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")
    return

def test_v_proj():
    W_filename = f"{pth_prefix}self_attn.v_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()

    w_zp_source_filename = f"{pth_prefix}self_attn.v_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()

    print(f"v_proj的数值范围:{torch.min(W)}, {torch.max(w_zp)}")
    print(f"v_proj的零点形状:{w_zp.shape}")
    print(f"v_proj的零点范围:{torch.min(w_zp)}, {torch.max(w_zp)}")

    A_out_true_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"v_proj输出数据形状是: {A_out_true.shape}")
    print(f"v_proj输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"v_proj输出零点形状是:{A_out_zp.shape}")
    print(f"v_proj输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")     # 备注: 这个zeropoint范围为-77~544, 需要clamp到[0, 255]
    return

def test_qkt_matmul():
    A_out_true_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"qkt_matmul输出数据形状是: {A_out_true.shape}")
    print(f"qkt_matmul输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"qkt_matmul输出零点形状是:{A_out_zp.shape}")
    print(f"qkt_matmul输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")  
    return

def test_softmax_out():
    A_out_true_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"Softmax输出数据形状是: {A_out_true.shape}")
    print(f"Softmax输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"Softmax输出零点形状是:{A_out_zp.shape}")
    print(f"Softmax输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")  

    # print(A_out_true[0, 0, 0:100])
    # print(A_out_true[0, 1, 0:100])
    # print(A_out_true[0, 2, 0:100])
    return

def test_pv_matmul_out():
    A_out_true_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"pv_matmul输出数据形状是: {A_out_true.shape}")
    print(f"pv_matmul输出范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"pv_matmul输出零点形状是:{A_out_zp.shape}")
    print(f"pv_matmul输出零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")  
    return

def test_out_proj():
    W_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()

    w_zp_source_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()

    print(f"out_proj的数值范围:{torch.min(W)}, {torch.max(w_zp)}")
    print(f"out_proj的零点形状:{w_zp.shape}")
    print(f"out_proj的零点范围:{torch.min(w_zp)}, {torch.max(w_zp)}")

    A_out_true_filename = f"{pth_prefix}final_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    A_out_zp_filename = f"{pth_prefix}final_layer_norm.in_quantizer/zeropoint.pth"
    A_out_zp = torch.load(A_out_zp_filename, map_location=torch.device('cpu')).detach()

    print(f"下一个block的输入数据形状是: {A_out_true.shape}")
    print(f"下一个block的输入范围是:{torch.min(A_out_true)}, {torch.max(A_out_true)}")
    print(f"下一个block的输入零点形状是:{A_out_zp.shape}")
    print(f"下一个block的输入零点范围是:{torch.min(A_out_zp)}, {torch.max(A_out_zp)}")
    return


def main():
    # 0. layernorm的输入
    test_layernorm_in()

    # 1. layernorm的输出
    test_layernorm_out()

    # 2. Q矩阵的输出
    test_q_proj()

    # 3. K矩阵的输出
    test_k_proj()

    # 4. V矩阵的输出
    test_v_proj()

    # 5. qkt_matmul的输出
    test_qkt_matmul()

    # 6. softmax的输出
    test_softmax_out()

    # 7. pv_matmul的输出
    test_pv_matmul_out()

    # 8. out_proj的输出
    test_out_proj()
    return 


if __name__ == "__main__":
    main()