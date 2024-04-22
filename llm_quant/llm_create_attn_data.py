"""
参数化虚考虑:
1. FP的参数在存的时候都不需要补0.因为weight_bram一行只能存256个数。FP参数在存的时候都是256的整数倍个。
2. linear参数在存的时候需要考虑在某些行的后面补0.

"""

import torch
import numpy as np
from transfer_data import *
import math
from llm_create_fc_data import (data2hex, 
                            npy2hex, 
                            npy2hex2D,
                            trans_W_to_bitW,
                            reorder_fp_data,
                            little_head) 
from llm_args import args
data_dir = args.data_dir_attn + "saved_data/"
pth_prefix = args.pth_prefix
HEIGHT_BLOCK = args.hidden_size // args.array_size
WIDTH_BLOCK = args.hidden_size // args.array_size


def create_fp0_data():
    # 只有layer 0的attn_layernorm需要这个quantizer参数, 因为
    # embedding的输出是FP, 这里需要先将它变成int8.
    # 其他layer的输入直接就被上一层处理为int8了。
    # 或许embedding输出的量化不用硬件完成？
    return


def create_nonlinear_type0_data():
    # Layernorm部分的非线性常数 
    QM = 16
    Qlog_d = 85         # 16'b0000_0000_0101_0101
    QM_hex = data2hex(QM, 16, 0)
    Qlog_d_hex = data2hex(Qlog_d, 16, 0)

    len_const = 7
    total_length = int(args.weight_bram_width / 16)
    zeros = torch.zeros(size=(total_length-len_const,), dtype=torch.int16)
    npy2hex(zeros, data_dir+"nonlinear_type0_data.txt", 16, 0, num_per_line=total_length, type='w')
    with open(data_dir+"nonlinear_type0_data.txt", 'a') as f:
        f.write(Qlog_d_hex)
        f.write('0000')
        f.write('00000000')
        f.write('0000')
        f.write(Qlog_d_hex)
        f.write(QM_hex)
        f.write("\n")
    return 


def create_layernorm_xin():
    # 准备输入给Layernorm层的X，都是8bit数据
    # [2048, 768]
    # 存后的文件大小:64*8/4=128列，2048*12=24576列。
    X_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    X = torch.load(X_filename, map_location=torch.device('cpu')).detach()
    X = X.squeeze()
    npy2hex2D(torch.round(X).numpy(), data_dir+"Layernorm_Xin.txt", args.x_bit, 0, args.array_size, "w")
    return


def create_layernorm_out_fp_data():
    # layernorm输出的int32需要经过FP模块变成int8
    # 这里提供量化的参数FPW和FPB。
    # 各有768个数，每行存256个数，一共存768/256*2=6行数据。
    # 算法上实现的layernorm的输出是uint8, 我这里将bias减去128, 
    # 这样就可以使layernorm的输出为int8.
    ln_wd2m_source_filename = f"{pth_prefix}self_attn_layer_norm/w_div_2M.pth"
    ln_wd2m = torch.load(ln_wd2m_source_filename, map_location='cpu').detach()

    ln_bias_filename = f"{pth_prefix}self_attn_layer_norm/bias.pth"
    ln_bias = torch.load(ln_bias_filename, map_location=torch.device('cpu')).detach()   

    scale_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zero_point_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
    zero_point = torch.load(zero_point_filename, map_location=torch.device('cpu')).detach()

    ln_wd2m_fusion = ln_wd2m / scale
    ln_bias_fusion = ln_bias / scale + zero_point - args.x_shift
    reorderd_fp_data = reorder_fp_data(ln_wd2m_fusion, ln_bias_fusion).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"Layernorm_out_fp.txt", num_per_line=args.num_per_line_fp, type="w")
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"Layernorm_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    
    return 


def create_weight(name="q_proj"):
    # 四个权重矩阵的大小都是768*768
    # 都是2bit无符号数据
    # Bram中每一行存一个64*64*1bit的块
    # 一共存12*12*2=288行
    # 重要问题:  从pth文件读取的W实际上还是按照output_channel * input_channel的方式
    # 存储的, 所以我需要先将它转置一下。
    NUM_PER_LINE = args.weight_bram_width // 4
    W_filename = f"{pth_prefix}self_attn.{name}.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()

    W = torch.round(W).numpy().T
    W_1bit_compress = trans_W_to_bitW(W, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    npy2hex(W_1bit_compress, data_dir+name+"_weight.txt", total_bits=4, frac_bits=0, num_per_line=NUM_PER_LINE)
    return 


def create_linear(name="q_proj"):
    # 存每个linear层的Wz, Xz.
    # 其中Wz是2bit有符号数据，共768个.但我要把它们当成4bit。
    # Xz是8bit有符号数据，共768个。
    # 它们都被存为4096bit一行。
    # x_z都要偏移。
    NUM_PER_LINE1 = args.weight_bram_width // args.x_bit
    NUM_PER_LINE2 = args.weight_bram_width // args.wz_bit
    BLOCK_PER_LINE1 = args.weight_bram_width // (args.x_bit * args.array_size)
    BLOCK_PER_LINE2 = args.weight_bram_width // (args.wz_bit * args.array_size)

    if name == "out_proj":
        x_zp_source_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
        x_zp = torch.load(x_zp_source_filename, map_location="cpu").detach()
    else:
        x_zp_source_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
        x_zp = torch.load(x_zp_source_filename, map_location="cpu").detach()

    x_zp = x_zp - args.x_shift

    x_zp = x_zp.reshape(-1,args.array_size) 
    zero_block = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1) * BLOCK_PER_LINE1 - HEIGHT_BLOCK
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    x_zp = torch.cat((x_zp, zeros), dim=0)
    x_zp = torch.round(x_zp).numpy()

    w_zp_source_filename = f"{pth_prefix}self_attn.{name}.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()
    w_zp = w_zp.reshape(-1, args.array_size)
    zero_block = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2) * BLOCK_PER_LINE2 - WIDTH_BLOCK
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    w_zp = torch.cat((w_zp, zeros), dim=0)
    w_zp = torch.round(w_zp).numpy()

    npy2hex2D(x_zp, data_dir+name+"_linear.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir+name+"_linear.txt", total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")
    
    # 小头存储
    x_zp_saved_line = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1)
    x_zp = little_head(x_zp, x_zp_saved_line, type="xz")
    w_zp_saved_line = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2)
    w_zp = little_head(w_zp, w_zp_saved_line, type="wz")
    npy2hex2D(x_zp, data_dir + name + "_linear_v2.txt",total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir + name + "_linear_v2.txt",total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")
    return


def create_qkv_out_fp_data(name="q_proj"):
    # q_proj, k_proj, v_porj模块后面的quantizer参数
    # 注意要进行参数融合。特别要注意对于q_proj来说，还要考虑除以根号d的操作。
    # 大小：FPW和FPB都是长为768的浮点数。
    # 每行存256个数，一共存6行。
    Sx1_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/scale.pth"
    Sx1 = torch.load(Sx1_filename, map_location='cpu').detach()

    Sw1_filename = f"{pth_prefix}self_attn.{name}.weight_quantizer/scale.pth"
    Sw1 = torch.load(Sw1_filename, map_location=torch.device('cpu')).detach()     

    if name == "q_proj":
        Sx2_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/scale.pth"
        Zp2_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    elif name == "k_proj":
        Sx2_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/scale.pth"
        Zp2_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    elif name == "v_proj":
        Sx2_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/scale.pth"
        Zp2_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/zeropoint.pth"
    Sx2 = torch.load(Sx2_filename, map_location=torch.device('cpu')).detach()
    Zp2 = torch.load(Zp2_filename, map_location=torch.device('cpu')).detach()

    bias_filename = f"{pth_prefix}self_attn.{name}/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    if name == "q_proj":
        FPW = Sx1 * Sw1 / Sx2 / math.sqrt(args.head_size)
        FPB = torch.round(Zp2 - args.x_shift) + bias / Sx2 / math.sqrt(args.head_size)
    else: 
        FPW = Sx1 * Sw1 / Sx2
        FPB = torch.round(Zp2) + bias / Sx2
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data), data_dir+name+"_out_fp.txt",num_per_line=args.num_per_line_fp, type='w')
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+name+"_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    return


def create_qkt_linear():
    # 存储matmul(Q, K.T)的两个zeropoint: Q_zp, K_zp.
    # 二者都是1个数，需要复制成kv cache长度个数。Q相当于act，K相当于weight。
    # 其中，前者需要减去128进行偏移。
    # 问题：复制多少份合适？KV cache最大长度为2048，所以应该复制2048份。
    NUM_PER_LINE = args.weight_bram_width // args.x_bit
    Q_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    Q_zp = torch.load(Q_zp_filename, map_location=torch.device('cpu')).detach() - args.x_shift
    Q_zp = Q_zp.expand(args.vector_block)

    K_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    K_zp = torch.load(K_zp_filename, map_location=torch.device('cpu')).detach()  # 应该是一个数
    K_zp = K_zp.expand(args.vector_block)

    zero_len = math.ceil(args.vector_block / NUM_PER_LINE) * NUM_PER_LINE - args.vector_block
    zeros = torch.zeros(size=(zero_len,), dtype=torch.int8)

    Q_zp = torch.cat((Q_zp, zeros), dim=0).numpy()
    K_zp = torch.cat((K_zp, zeros), dim=0).numpy()

    npy2hex(Q_zp, data_dir + "qkt_linear.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="w")
    npy2hex(K_zp, data_dir + "qkt_linear.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="a")
    
    # 小头存储
    # 由于这些数都长一样，所以不需要小头存储
    # 我做个假的小头
    npy2hex(Q_zp, data_dir + "qkt_linear_v2.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="w")
    npy2hex(K_zp, data_dir + "qkt_linear_v2.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="a")
    return

def create_qkt_out_fp_data():
    # Q * K.T矩阵乘之后的quantizer参数
    # 大小：FPW和FPB都是长为1的数，不过为了和之前的fp模块保持一致，我把它们都
    # 复制成为768的浮点数。
    # 每行存256个数，一共存6行。
    Sq_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/scale.pth"
    Sq = torch.load(Sq_filename, map_location='cpu').detach()

    Sk_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/scale.pth"
    Sk = torch.load(Sk_filename, map_location='cpu').detach()

    Sout_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/scale.pth"
    Sout = torch.load(Sout_filename, map_location='cpu').detach()

    Zp_out_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/zeropoint.pth"
    Zp_out = torch.load(Zp_out_filename, map_location='cpu').detach()

    FPW = Sq * Sk / Sout
    FPB = torch.round(Zp_out)
    FPW = FPW.expand(HEIGHT_BLOCK * args.array_size)
    FPB = FPB.expand(HEIGHT_BLOCK * args.array_size)
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data), data_dir+"qkt_out_fp.txt",num_per_line=args.num_per_line_fp, type='w')
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"qkt_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    return


def create_nonlinear_type1_data():
    # Softmax部分的非线性常数
    # 6个非线性常数，占了112bit(有3个是layernorm的，所以写为0)
    # 左边的bit都补零
    # t1 = Qs_div_ln2
    # t2 = Qln2_div_s
    # t3 = Q2_div_s
    t1_source_filename = f"{pth_prefix}self_attn.softmax/t1.pth"
    t1 = torch.load(t1_source_filename, map_location=torch.device('cpu')).item()
    t1_hex = data2hex(t1, 16, 16)                                                  

    t2_source_filename = f"{pth_prefix}self_attn.softmax/t2.pth" 
    t2 = torch.load(t2_source_filename, map_location=torch.device('cpu')).item()                                
    t2_hex = data2hex(t2, 16, 8)                

    t3_source_filename = f"{pth_prefix}self_attn.softmax/t3.pth" 
    t3 = torch.load(t3_source_filename, map_location=torch.device('cpu')).item()
    t3_hex = data2hex(t3, 32, 8)                

    len_const = 7
    total_length = int(args.weight_bram_width / 16)
    zeros = torch.zeros(size=(total_length-len_const,), dtype=torch.int16)
    npy2hex(zeros, data_dir+"nonlinear_type1_data.txt", 16, 0, num_per_line=total_length, type='w')
    with open(data_dir+"nonlinear_type1_data.txt", 'a') as f:
        f.write('0000')
        f.write(t1_hex)       
        f.write(t2_hex)
        f.write(t3_hex)
        f.write('0000')
        f.write('0000')
        f.write("\n")
    return 


def create_softmax_out_fp_data():
    # 从softmax出来的数据不需要考虑softmax之前的quantizer的scale，
    # 因为softmax计算的分子分母会约掉这一项。
    # 所以根据ppt，FPW = 1/(2**M)/scale_out, FPB = 0 + zero_point - 128.
    # 也都复制vector_block的长度(2048)。
    scale_out_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/scale.pth"
    scale_out = torch.load(scale_out_filename, map_location="cpu").detach()

    zp_out_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/zeropoint.pth"
    zeropoint = torch.load(zp_out_filename, map_location="cpu").detach()

    M = args.softmax_m 
    FPW = 1 / 2 ** M / scale_out
    FPB = 0 - args.x_shift + zeropoint
    FPW = FPW.expand(args.vector_block)
    FPB = FPB.expand(args.vector_block)
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data), data_dir+"Softmax_out_fp.txt",num_per_line=args.num_per_line_fp, type='w')
    
    # 小头存储（其实所有数据都长一样，没必要小头）
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"Softmax_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    return


def create_pv_linear():
    # 存储matmul(p, v)的两个zeropoint: P_zp, V_zp.
    # 目前来说P_zp是一个数, 为0. 它需要先偏移128.
    # V_zp却是一个数组，768个数。
    # 都存8bit的话，P_zp存4行0；V_zp存2行，后面一行补一半的0。
    NUM_PER_LINE = args.weight_bram_width // args.x_bit
    P_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/zeropoint.pth"
    P_zp = torch.load(P_zp_filename, map_location=torch.device('cpu')).detach()
    P_zp = P_zp.expand(args.vector_block)
    zero_len = math.ceil(args.vector_block / NUM_PER_LINE) * NUM_PER_LINE - args.vector_block
    zeros = torch.zeros(size=(zero_len,), dtype=torch.int8)
    P_zp = torch.cat((P_zp, zeros), dim=0)
    P_zp = P_zp - args.x_shift
    P_zp = P_zp.numpy()

    V_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/zeropoint.pth"
    V_zp = torch.load(V_zp_filename, map_location=torch.device('cpu')).detach()
    zero_len = math.ceil(args.hidden_size / NUM_PER_LINE) * NUM_PER_LINE - args.hidden_size
    zeros = torch.zeros(size=(zero_len,), dtype=torch.int8)
    V_zp = torch.cat((V_zp, zeros), dim=0)
    V_zp = V_zp.numpy()

    npy2hex(P_zp, data_dir + "pv_linear.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="w")
    npy2hex(V_zp, data_dir + "pv_linear.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="a")

    # 小头存储
    P_zp_saved_line = len(P_zp) // NUM_PER_LINE
    V_zp_saved_line = len(V_zp) // NUM_PER_LINE
    P_zp = little_head(P_zp, P_zp_saved_line, type="xz")
    V_zp = little_head(V_zp, V_zp_saved_line, type="xz")
    npy2hex2D(P_zp, data_dir + "pv_linear_v2.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="w")
    npy2hex2D(V_zp, data_dir + "pv_linear_v2.txt", total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE, type="a")
    return

def create_pv_out_fp_data():
    # P * V 之后的quantizer参数
    # 大小: 这里Sx1是1个，但是Sx2是768个，Sx_out是1个，所以FPW长度768.
    # FPB长度为768。
    Sx1_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/scale.pth"
    Sx1 = torch.load(Sx1_filename, map_location='cpu').detach()

    Sx2_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/scale.pth"
    Sx2 = torch.load(Sx2_filename, map_location='cpu').detach()

    Sx_out_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    Sx_out = torch.load(Sx_out_filename, map_location='cpu').detach()

    Zp_out_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    Zp_out = torch.load(Zp_out_filename, map_location='cpu').detach()

    FPW = Sx1 * Sx2 / Sx_out
    FPB = torch.round(Zp_out) - args.x_shift
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data), data_dir+"pv_matmul_out_fp.txt",num_per_line=args.num_per_line_fp, type='w')
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"pv_matmul_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    return


def create_skip_s():
    # 在进行最后的残差连接时，layer_norm输入的int8需要乘以它的scale，
    # 这个scale就是skip_s.
    # 考虑到要融合FC block的Final_layernorm的in_quantizer, 所以它还要再除以s_next_block.
    s_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    s = torch.load(s_filename, map_location=torch.device('cpu')).item()

    s_final_layernorm_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    s_next_block = torch.load(s_final_layernorm_filename, map_location=torch.device('cpu')).item()

    skip = np.array(s / s_next_block).astype(np.float16)
    bin = fp16_to_binary(skip)
    s_hex = bin2hex(bin, 4)

    len_const = 8
    total_length = int(args.weight_bram_width // 16)
    zeros = torch.zeros(size=(total_length-len_const,), dtype=torch.int16)
    npy2hex(zeros, data_dir+"skip_data.txt", 16, 0, num_per_line=total_length, type='w')
    with open(data_dir+"skip_data.txt", 'a') as f:
        f.write(s_hex)
        for i in range(0,7):
            f.write("0000")
        f.write("\n")
    return


def create_o_out_fp_data():
    # 6行数据
    Sx_in_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    Sx_in = torch.load(Sx_in_filename, map_location=torch.device('cpu')).detach()

    Sw_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/scale.pth"
    Sw = torch.load(Sw_filename, map_location=torch.device('cpu')).detach()  

    Sx_out_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    Sx_out = torch.load(Sx_out_filename, map_location=torch.device('cpu')).detach()

    Zp_out_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/zeropoint.pth"
    Zp_out = torch.load(Zp_out_filename, map_location=torch.device('cpu')).detach()

    bias_filename = f"{pth_prefix}self_attn.out_proj/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    Sx_next_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    Sx_next = torch.load(Sx_next_filename, map_location=torch.device('cpu')).detach()

    FPW = Sx_in * Sw / Sx_next
    FPB = bias / Sx_next + Sx_out * Zp_out / Sx_next
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()

    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"out_proj_out_fp.txt", num_per_line=args.num_per_line_fp, type="w")
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // args.num_per_line_fp)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"out_proj_out_fp_v2.txt", num_per_line=args.num_per_line_fp, type="w")
    return 


def trans_txt_to_mem(file_in, file_list):
    with open(file_in, 'r') as txt:
        with open(data_dir+"Layernorm_xin.mem", 'w') as mem:
            address = 0
            for line in txt:
                mem.write(f'@{address:08x} {line}')
                address += 1
    
    address = 0
    with open(data_dir + "Bram_data.mem", "w") as mem:
        for file in file_list:
            with open(file, 'r') as txt:
                for line in txt:
                    mem.write(f'@{address:08x} {line}')
                    address += 1

    with open(data_dir + "Bram_data.txt","w") as f:
        for file in file_list:
            with open(file, 'r') as txt:
                for line in txt:
                    f.write(line)

    # 后来发现其实不需要reorder, 对于一个bram来说, 如果它有两个端口, 一个宽度是4096， 一个
    # 宽度是1024, 那么默认将初始化的mem文件按照短口写入。即使mem文件有4096列，那也只会截取低位的
    # 1024列写入。所以我现在要做的就是将4096列*2354行的Bram_data.txt文件里面的数据转为1024列*9416行
    # 的数据。对于原来的每一行，先存低位。
    address = 0
    with open(data_dir+"Long_bram_data.mem", "w") as mem,\
        open (data_dir+"Long_bram_data.txt", "w") as txt:
        with open(data_dir+"Bram_data.txt", "r") as f:
            for line in f:
                row = line.strip()
                row3 = row[0:256]
                row2 = row[256:512]
                row1 = row[512:768]
                row0 = row[768:1024]
                mem.write(f'@{address:08X} {row0}\n')
                address += 1
                mem.write(f'@{address:08X} {row1}\n')
                address += 1
                mem.write(f'@{address:08X} {row2}\n')
                address += 1
                mem.write(f'@{address:08X} {row3}\n')
                address += 1
                txt.write(f'{row0}\n')
                txt.write(f'{row1}\n')
                txt.write(f'{row2}\n')
                txt.write(f'{row3}\n')
    return 


def llm_create_attn_data():
    # 0. 只有layer0才有的: 对输入的embedding需要量化一下
    # 不过这可以交给cpu做。也就是说FPGA上面每一层的输入都是int8
    # create_fp0_data()

    # 1.Attention模块的layernorm的非线性数据
    create_nonlinear_type0_data()

    # 2. Layernorm的输入数据（对于layer 0来说是不是不需要？）
    create_layernorm_xin()

    # 3.layernorm之后的FP数据(attn_layer_norm.in_quantizer)
    create_layernorm_out_fp_data()

    # 4. Wq weight
    create_weight(name="q_proj")

    # 5. Wq的-Wz, -Xz
    create_linear(name="q_proj")

    # 6. Wk_weight
    create_weight(name="k_proj")

    # 7. Wk的-Wz, -Xz
    create_linear(name="k_proj")

    # 8. Wv_weight
    create_weight(name="v_proj")

    # 9. Wv的-Wz, -Xz
    create_linear(name="v_proj")

    # 10. Wq之后的FP数据(qkt_matmul.x1_quantizer): int32->int8
    create_qkv_out_fp_data(name="q_proj")

    # 11. Wk之后的FP数据(qkt_matmul.x2_quantizer): int32->int8
    create_qkv_out_fp_data(name="k_proj")

    # 12. qkt_matmul计算时的linear数据
    create_qkt_linear()

    # 13. qkt计算之后的FP数据(softmax.in_quanziter): int32->int8
    create_qkt_out_fp_data()

    # 14. Softmax计算需要的非线性数据
    create_nonlinear_type1_data()

    # 15. softmax之后的FP数据(pv_matmul.x1_quantizer)
    create_softmax_out_fp_data()

    # 16. Wv之后的FP数据(pv_matmul.x2_quantizer)
    create_qkv_out_fp_data(name="v_proj")

    # 17. pv_matmul计算时的linear数据
    create_pv_linear()

    # 18. Wo之前的FP数据(out_proj.act_quantizer)
    create_pv_out_fp_data()

    # 19. Wo weight
    create_weight(name="out_proj")

    # 20. Wo的-Wz, -Xz
    create_linear(name="out_proj")

    # 21. 遥远的skip_s
    create_skip_s()

    # 22. Wo之后的FP数据(计算后得到layers.0.final_layer_norm的int8输入) 
    create_o_out_fp_data()

    # 创建mem文件
    # 先有一个总的txt文件。
    # 然后有输入给attention block的int 8的mem文件。
    # 所有其他19个txt转化给weight_bram的mem文件。一共是1215行。
    # weight_bram_reorder文件。
    # 所有的linear, fp数据都用v2文件。
    # 因为v1文件我是单纯用来验证的。
    file_in = data_dir+'Layernorm_Xin.txt'
    weight_bram_file_list = [data_dir+"nonlinear_type0_data.txt",       # 1     lines
                             data_dir+"nonlinear_type1_data.txt",       # 1     lines
                             data_dir+"Layernorm_out_fp.txt",           # 6     lines
                             data_dir+"q_proj_weight.txt",              # 288   lines
                             data_dir+"q_proj_linear_v2.txt",           # 3     lines
                             data_dir+"q_proj_out_fp_v2.txt",           # 6     lines
                             data_dir+"k_proj_weight.txt",              # 288   lines
                             data_dir+"k_proj_linear_v2.txt",           # 3     lines
                             data_dir+"k_proj_out_fp_v2.txt",           # 6     lines
                             data_dir+"v_proj_weight.txt",              # 288   lines
                             data_dir+"v_proj_linear_v2.txt",           # 3     lines
                             data_dir+"v_proj_out_fp_v2.txt",           # 6     lines
                             data_dir+"qkt_linear_v2.txt",              # 8     lines
                             data_dir+"qkt_out_fp_v2.txt",              # 6     lines
                             data_dir+"Softmax_out_fp_v2.txt",          # 6     lines
                             data_dir+"pv_linear_v2.txt",               # 6     lines
                             data_dir+"pv_matmul_out_fp_v2.txt",        # 6     lines
                             data_dir+"out_proj_weight.txt",            # 288   lines
                             data_dir+"out_proj_linear_v2.txt",         # 3     lines
                             data_dir+"skip_data.txt",                  # 1     lines
                             data_dir+"out_proj_out_fp_v2.txt"          # 6     lines
                             ]
    trans_txt_to_mem(file_in=file_in, file_list=weight_bram_file_list)
    return 


if __name__ == '__main__':
    llm_create_attn_data()