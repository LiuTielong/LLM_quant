import torch
import numpy as np
import math
import struct

from llm_test_fc_data import (
    Matmul_compute,
    )

from llm_args import args
data_dir = args.data_dir_attn
pth_prefix = args.pth_prefix


def test_q_proj_simple():
    W_filename = f"{pth_prefix}self_attn.q_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()
    W = torch.round(W)
    w_zp_source_filename = f"{pth_prefix}self_attn.q_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()
    weight_scale_filename = f"{pth_prefix}self_attn.q_proj.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_filename, map_location=torch.device('cpu')).detach()
    W_shift = W.T - w_zp                    # 一定要记得转置！

    bias_filename = f"{pth_prefix}self_attn.q_proj/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    Xin_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/x_int_add_zp_clamp.pth"
    Xin = torch.load(Xin_filename, map_location=torch.device('cpu')).detach()
    In_scale_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/scale.pth"
    In_scale = torch.load(In_scale_filename, map_location=torch.device('cpu')).detach()
    x_zp_source_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
    x_zp = torch.load(x_zp_source_filename, map_location="cpu").detach()
    Xin_shift = Xin - x_zp

    Out_scale_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/scale.pth"
    Out_scale = torch.load(Out_scale_filename, map_location=torch.device('cpu')).detach()

    Out_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    Out_zp = torch.load(Out_zp_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)
    A_out = torch.clamp(torch.round((Matmul_out * weight_scale * In_scale + bias) / Out_scale/8 + Out_zp), 0, 255)  # 这里要记得除以8！

    A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    A_out_true = A_out_true.squeeze()

    print(A_out[0,0,0:100])
    print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("对不上的数据个数为:", different_pos.shape) 
    """
    实验结论: 只有11个数对不上, 且相差都是1.这是可以接受的。
    十分值得注意的是W的计算得先转置, 
    以及hidden_states经过q_proj后还有一个除以8的scaling操作。
    """
    return 

def test_k_proj_simple():
    W_filename = f"{pth_prefix}self_attn.k_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()
    W = torch.round(W)
    w_zp_source_filename = f"{pth_prefix}self_attn.k_proj.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()
    weight_scale_filename = f"{pth_prefix}self_attn.k_proj.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_filename, map_location=torch.device('cpu')).detach()
    W_shift = W.T - w_zp

    bias_filename = f"{pth_prefix}self_attn.k_proj/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    Xin_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/x_int_add_zp_clamp.pth"
    Xin = torch.load(Xin_filename, map_location=torch.device('cpu')).detach()
    In_scale_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/scale.pth"
    In_scale = torch.load(In_scale_filename, map_location=torch.device('cpu')).detach()
    x_zp_source_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
    x_zp = torch.load(x_zp_source_filename, map_location="cpu").detach()
    Xin_shift = Xin - x_zp

    Out_scale_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/scale.pth"
    Out_scale = torch.load(Out_scale_filename, map_location=torch.device('cpu')).detach()

    Out_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    Out_zp = torch.load(Out_zp_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)
    A_out = torch.clamp(torch.round((Matmul_out * weight_scale * In_scale + bias) / Out_scale + Out_zp), 0, 255)

    A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    A_out_true = A_out_true.squeeze()

    print(A_out[0,0,0:100])
    print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("对不上的数据个数为:", different_pos.shape) 
    return 

def test_qkt_matmul_out_simple():
    NUM_OF_HEAD = 12
    HEAD_DIM = 64
    # 1. 读取Q和K, 读取它们各自的Zp
    Q_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    Q = torch.load(Q_filename, map_location=torch.device('cpu')).detach().squeeze()

    K_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    K = torch.load(K_filename, map_location=torch.device('cpu')).detach().squeeze()

    Q_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    Q_zp = torch.load(Q_zp_filename, map_location=torch.device('cpu')).detach() # 1个数

    K_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    K_zp = torch.load(K_zp_filename, map_location=torch.device('cpu')).detach()  # 1个数

    # 2. 进行矩阵乘法
    # 它们都是8bit无符号数, 所以还是能做bit-stream的矩阵乘。
    # 这里把Q当作X, K当作W.
    Q = torch.round(Q)
    K = torch.round(K)
    Q = Q.view(args.vector_block, NUM_OF_HEAD, HEAD_DIM).transpose(0, 1)
    K = K.view(args.vector_block, NUM_OF_HEAD, HEAD_DIM).transpose(0, 1)
    KT = K.transpose(1, 2)
    Matmul_out = np.matmul(Q - Q_zp, KT - K_zp)                           # [12, 2048, 2048]

    # 3. 进行矩阵乘之后的FP单元量化
    # 下面这四个值都是一个数，太好了
    Sq_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/scale.pth"
    Sq = torch.load(Sq_filename, map_location=torch.device('cpu')).detach()

    Sk_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/scale.pth"
    Sk = torch.load(Sk_filename, map_location=torch.device('cpu')).detach()

    Sout_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/scale.pth"
    Sout = torch.load(Sout_filename, map_location=torch.device('cpu')).detach()

    Zpout_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/zeropoint.pth"
    Zpout = torch.load(Zpout_filename, map_location=torch.device('cpu')).detach()

    A_out = torch.clamp(torch.round(Matmul_out * Sq * Sk / Sout + Zpout), -128, 127)
    print(A_out[0, 0, 0:100])

    # 4. 加载真实值进行比较
    A_out_true_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    print(A_out_true[0, 0, 0:100])
    diffrent_pos = torch.nonzero(A_out != A_out_true)
    print(diffrent_pos.shape)
    """
    实验结论:仅仅有2个位置不同, 所以这些数据的准确性还是很高的。
    值得一提的是这里量化的数据又是有符号的了。因为Layernorm和Softmax的输入是int8, 矩阵乘的输入都是uint8.
    """
    return 

def test_pv_matmmul_out_simple():
    print("开始测试pv_matmul的输出。")
    P_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    P = torch.load(P_filename, map_location=torch.device('cpu')).detach()

    V_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    V = torch.load(V_filename, map_location=torch.device('cpu')).detach()

    P_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/zeropoint.pth"
    P_zp = torch.load(P_zp_filename, map_location=torch.device('cpu')).detach() - args.x_shift

    V_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/zeropoint.pth"
    V_zp = torch.load(V_zp_filename, map_location=torch.device('cpu')).detach()

    P = torch.round(P) - args.x_shift
    V = torch.round(V)
    V = V.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)

    P = P.numpy()
    V = V.numpy()
    P_zp = P_zp.numpy()
    V_zp = V_zp.numpy()

    Matmul_out = np.zeros((args.num_head, args.vector_block, args.head_size))
    for head_index in range(args.num_head):
        head_V = V[head_index]
        head_Vzp = V_zp[head_index*args.head_size:(head_index+1)*args.head_size].reshape(1, args.head_size)
        head_P = P[head_index]
        head_Pzp = P_zp
        Matmul_out[head_index] = Matmul_compute(
            head_V, head_Vzp, head_P, head_Pzp,
            VECTOR_BLOCK=args.vector_block,
            HEIGHT_BLOCK=args.vector_block // args.array_size,
            WIDTH_BLOCK=args.head_size // args.array_size,
            BIT_BLOCK=args.x_bit
        )
    Matmul_out = torch.tensor(np.round(Matmul_out))
    Matmul_out = Matmul_out.transpose(0, 1).reshape(args.vector_block, -1)

    Sp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/scale.pth"
    Sp = torch.load(Sp_filename, map_location=torch.device('cpu')).detach()

    Sv_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/scale.pth"
    Sv = torch.load(Sv_filename, map_location=torch.device('cpu')).detach()

    Sout_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    Sout = torch.load(Sout_filename, map_location=torch.device('cpu')).detach()

    Zpout_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    Zpout = torch.load(Zpout_filename, map_location=torch.device('cpu')).detach() - args.x_shift

    A_out = torch.clamp(torch.round(Matmul_out * Sp * Sv / Sout + Zpout), -128, 127)
    print(A_out[0,0:100])

    # 加载真实值进行比较
    A_out_true_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach() - args.x_shift
    A_out_true = torch.round(A_out_true[0])
    print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("不相等的数据个数为:", different_pos.shape)
    # 实验结论: 仅仅一个数不相等
    return

def test_out_proj_simple():
    # 主要是有一个残差连接
    W_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location="cpu").detach()
    W = torch.round(W)
    W_zp_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/zeropoint.pth"
    W_zp = torch.load(W_zp_filename, map_location="cpu").detach()
    W_scale_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/scale.pth"
    W_scale = torch.load(W_scale_filename, map_location="cpu").detach()
    Bias_filename = f"{pth_prefix}self_attn.out_proj/bias.pth"
    Bias = torch.load(Bias_filename, map_location="cpu").detach()
    W_shift = W.T - W_zp

    Xin_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/x_int_add_zp_clamp.pth"
    Xin = torch.load(Xin_filename, map_location="cpu").detach()
    In_scale_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    In_scale = torch.load(In_scale_filename, map_location="cpu").detach()
    X_zp_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    X_zp = torch.load(X_zp_filename, map_location="cpu").detach()
    Xin_shift = Xin - X_zp


    Sx_ln_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    Sx_ln = torch.load(Sx_ln_filename, map_location=torch.device('cpu')).detach()
    Zp_ln_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/zeropoint.pth"
    Zp_ln = torch.load(Zp_ln_filename, map_location=torch.device('cpu')).detach()

    Sx_next_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    Sx_next = torch.load(Sx_next_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)

    LN_Xin_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    LN_Xin = torch.load(LN_Xin_filename, map_location="cpu").detach()

    FPW = In_scale * W_scale / Sx_next
    FPB = Bias / Sx_next + Sx_ln * Zp_ln / Sx_next
    skip = Sx_ln / Sx_next
    A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB + LN_Xin * skip), -128, 127)
    print(A_out[0,0,0:100])

    # 读取真实输出
    A_out_true_filename = f"{pth_prefix}final_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location="cpu").detach()
    print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("不相等的数据个数为:", different_pos.shape)
    # 结论: 只有5个数不正确，表示这个计算流程是没问题的。
    return

def llm_test_attn_data_simple():
    # test_q_proj_simple()
    # test_k_proj_simple()
    # test_qkt_matmul_out_simple()
    # test_pv_matmmul_out_simple()
    # test_out_proj_simple()
    return

if __name__ == "__main__":
    llm_test_attn_data_simple()