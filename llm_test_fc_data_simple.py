import torch
import numpy as np
import math
import struct
from llm_create_fc_data import *
import pdb
from llm_args import args

data_dir = args.data_dir_fc
pth_prefix = args.pth_prefix
pth_next_prefix = args.pth_next_prefix


def test_fc1_out_simple():
    # 1. 浮点数运算结果
    weight_dequant_source_filename = f"{pth_prefix}fc1.weight_quantizer/dequant.pth"
    weight_dequant = torch.load(weight_dequant_source_filename, map_location=torch.device('cpu')).detach()

    bias_filename = f"{pth_prefix}fc1/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    x_dequant_filename = f"{pth_prefix}final_layer_norm.out_quantizer/dequant.pth"
    x_dequant = torch.load(x_dequant_filename, map_location=torch.device('cpu')).detach()             

    X_out_dequant = torch.nn.functional.linear(x_dequant, weight_dequant) + bias        

    A_out_true_filename = f"{pth_prefix}fc1/fp_out.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()

    def mse_loss(x, y):
        return torch.mean((x - y) ** 2) > 1e-2
    dif_pos = mse_loss(X_out_dequant, A_out_true)
    print(torch.sum(dif_pos==True))
    # 结论：浮点数运算结果没问题， X_out_dequant = A_out_true


    # 2. 模拟整数运算
    # 我这里做偏移128之后的验证
    weight_scale_source_filename = f"{pth_prefix}fc1.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_source_filename, map_location=torch.device('cpu'))

    weight_zp_source_filename = f"{pth_prefix}fc1.weight_quantizer/zeropoint.pth"
    weight_zp = torch.load(weight_zp_source_filename, map_location=torch.device('cpu'))

    W = (weight_dequant.T / weight_scale + weight_zp) 
    W_shift = W - weight_zp

    bias_filename = f"{pth_prefix}fc1/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    dequant_filename = f"{pth_prefix}final_layer_norm.out_quantizer/dequant.pth"
    In_dequant = torch.load(dequant_filename, map_location=torch.device('cpu')).detach()

    scale_filename = f"{pth_prefix}final_layer_norm.out_quantizer/scale.pth"
    In_scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zeropoint_filename = f"{pth_prefix}final_layer_norm.out_quantizer/zeropoint.pth"
    In_zeropoint = torch.load(zeropoint_filename, map_location=torch.device('cpu')).detach()

    Xin = torch.round(In_dequant / In_scale + In_zeropoint)
    Xin_shift = Xin - In_zeropoint

    Out_scale_filename = f"{pth_prefix}fc2.act_quantizer/scale.pth"
    Out_scale = torch.load(Out_scale_filename, map_location=torch.device('cpu')).detach()

    Out_zp_filename = f"{pth_prefix}fc2.act_quantizer/zeropoint.pth"
    Out_zp = torch.load(Out_zp_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)
    Matmul_out = torch.clamp(torch.round((Matmul_out * weight_scale * In_scale + bias) / Out_scale + Out_zp - 128), -128, 127)
    Matmul_out = Matmul_out.detach()

    True_out_filename = f"{pth_prefix}fc2.act_quantizer/x_int_add_zp_clamp.pth"
    True_out = torch.load(True_out_filename, map_location=torch.device('cpu')).detach() - 128

    different_pos = torch.nonzero(True_out != Matmul_out).squeeze()
    print("不相等的数据个数为: ",different_pos)
    # pdb.set_trace()
    # 结果：有2个数据出现偏差，差距应该都在+-1。
    # 不过这点小事我也就先不管了。
    return 


def test_fc2_out_simple():
    # 1. 浮点数运算结果
    # 1.1 先进行矩阵乘运算
    weight_dequant_source_filename = f"{pth_prefix}fc2.weight_quantizer/dequant.pth"
    weight_dequant = torch.load(weight_dequant_source_filename, map_location=torch.device('cpu')).detach() 

    bias_filename = f"{pth_prefix}fc2/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()                       

    x_dequant_filename = f"{pth_prefix}fc2.act_quantizer/dequant.pth"
    x_dequant = torch.load(x_dequant_filename, map_location=torch.device('cpu')).detach()                  

    X_out_dequant = torch.nn.functional.linear(x_dequant, weight_dequant) + bias                         

    X_out_true_filename = f"{pth_prefix}fc2/fp_out.pth"
    X_out_true = torch.load(X_out_true_filename, map_location=torch.device('cpu')).detach()

    # print(X_out_dequant[8, 25:45])
    # print(X_out_true[8, 25:45])
    # 结果：二者完全一样

    # 1.2 然后进行残差连接
    X0_filename = f"{pth_prefix}final_layer_norm.in_quantizer/dequant.pth"
    X0 = torch.load(X0_filename, map_location=torch.device('cpu')).detach()

    Layer2_out_filename = f"{pth_prefix}final_layer_norm.in_quantizer2/dequant.pth"
    Layer2_out = torch.load(Layer2_out_filename, map_location=torch.device('cpu')).detach()

    Block_out_before_quant = X_out_dequant + X0
    Block_out = Layer2_out + X0

    Block_out_true_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/input.pth"
    Block_out_true = torch.load(Block_out_true_filename, map_location=torch.device('cpu')).detach()

    # print(Block_out_before_quant[8, 20:40])
    # print(Block_out[8, 20:40])
    # print(Block_out_true[0, 8, 20:40])
    # 结果: 后两组数据完全相等, 第一组数据和它们能匹配上，乃是fake quant之前的版本


    # 2. 模拟整数运算结果
    # 2.1 矩阵乘部分的结果模拟
    weight_scale_source_filename = f"{pth_prefix}fc2.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_source_filename, map_location=torch.device('cpu'))

    weight_zp_source_filename = f"{pth_prefix}fc2.weight_quantizer/zeropoint.pth"
    weight_zp = torch.load(weight_zp_source_filename, map_location=torch.device('cpu'))

    W = (weight_dequant.T / weight_scale + weight_zp)     
    W_shift = W - weight_zp

    bias_filename = f"{pth_prefix}fc2/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    dequant_filename = f"{pth_prefix}fc2.act_quantizer/dequant.pth"
    In_dequant = torch.load(dequant_filename, map_location=torch.device('cpu')).detach()

    scale_filename = f"{pth_prefix}fc2.act_quantizer/scale.pth"
    In_scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zeropoint_filename = f"{pth_prefix}fc2.act_quantizer/zeropoint.pth"
    In_zeropoint = torch.load(zeropoint_filename, map_location=torch.device('cpu')).detach()

    Xin = torch.round(In_dequant / In_scale + In_zeropoint)
    Xin_shift = Xin - In_zeropoint

    Out_scale_filename = f"{pth_prefix}final_layer_norm.in_quantizer2/scale.pth"
    Out_scale = torch.load(Out_scale_filename, map_location=torch.device('cpu')).detach()

    Out_zp_filename = f"{pth_prefix}final_layer_norm.in_quantizer2/zeropoint.pth"
    Out_zp = torch.load(Out_zp_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)
    Matmul_out = torch.clamp(torch.round((Matmul_out * weight_scale * In_scale + bias) / Out_scale + Out_zp), -128, 127)
    Matmul_out = Matmul_out.detach()

    True_matmul_out_filename = f"{pth_prefix}final_layer_norm.in_quantizer2/x_int_add_zp_clamp.pth"
    True_matmul_out = torch.load(True_matmul_out_filename, map_location=torch.device('cpu')).detach()
    True_matmul_out = torch.clamp(True_matmul_out, -128, 127)

    # print(Matmul_out[0, 0:20])
    # print(True_matmul_out[0, 0:20])

    different_pos = torch.nonzero(Matmul_out != True_matmul_out).squeeze()
    # 结论：这里又变成-128~127了，那也就都是8bit有符号数。
    # 但是FC1的输出还是8bit无符号数.
    # 现在仅仅有一个数据对不上，还是可以的。可以认为矩阵乘的运算正确了。


    # 2.2 加上残差连接
    # layernorm的in_quantizer的scale和in_quantizer2的scale是相等的
    LayerNorm_Xin_filename = f"{pth_prefix}final_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    LayerNorm_Xin = torch.load(LayerNorm_Xin_filename, map_location=torch.device('cpu')).detach()

    Layer1_scale_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    Layer1_scale = torch.load(Layer1_scale_filename, map_location=torch.device('cpu')).detach()

    Matmul_out = torch.matmul(Xin_shift, W_shift)
    Res_out = torch.clamp(torch.round((Matmul_out * weight_scale * In_scale + bias) / Out_scale + Out_zp), -128, 127) + LayerNorm_Xin
    Res_out = Res_out * Out_scale
    Layer1_in = torch.clamp(torch.round(Res_out / Layer1_scale), -128, 127)

    True_layer1_in_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    True_layer1_in = torch.load(True_layer1_in_filename, map_location=torch.device('cpu')).detach()

    different_pos = torch.nonzero(Layer1_in != True_layer1_in).squeeze()
    # print(different_pos.shape)
    # 结论：全部相等！


    # 2.3 算子融合后的结果试试:
    FPW = weight_scale * In_scale / Layer1_scale 
    FPB = bias / Layer1_scale + Out_scale / Layer1_scale * Out_zp
    skip = Out_scale / Layer1_scale
    Res_out_fusion = torch.clamp(torch.round(Matmul_out * FPW + FPB + LayerNorm_Xin * skip), -128, 127)

    different_pos = torch.nonzero(Res_out_fusion != True_layer1_in).squeeze()
    print(Res_out_fusion[0,0:100])
    print(True_layer1_in[0,0,0:100])
    print(different_pos.shape)
    """实验结论: 出现了323239个结果有偏差! 不过我们仍然要做."""
    return 


def llm_test_fc_data_simple():
    test_fc1_out_simple()
    test_fc2_out_simple()
    return

if __name__ == "__main__":
    llm_test_fc_data_simple()