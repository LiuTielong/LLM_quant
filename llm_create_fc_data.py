import torch
import numpy as np
from transfer_data import *
import math
from llm_args import args

data_dir = args.data_dir_fc
pth_prefix = args.pth_prefix
pth_next_prefix = args.pth_next_prefix

def fix2int(x, frac_bits):
    y = np.round(x*2**frac_bits)
    return y

def data2hex(data, total_bits, frac_bits):
    int_data = fix2int(data, frac_bits)
    bin_npy = int2bin(int_data, total_bits)
    return bin2hex(bin_npy, total_bits//4)

def npy2hex(pth_npy, target_filename, total_bits, frac_bits, num_per_line=1, type='w'):
    """将numpy数组转换为Vivado仿真可读取的hex文件"""
    num_count = 0
    int_npy = fix2int(pth_npy, frac_bits)
    with open(target_filename, type) as f:
        for num in int_npy:
            bin_npy = int2bin(num, total_bits)
            hex_npy = bin2hex(bin_npy, total_bits//4)
            if num_count == num_per_line-1:
                f.write(hex_npy + '\n')
                num_count = 0
            else:
                f.write(hex_npy)
                num_count += 1   

def npy2hex2D(pth_npy, target_filename, total_bits, frac_bits, num_per_line=1, type='w'):
    """将numpy数组转换为Vivado仿真可读取的hex文件"""
    num_count = 0
    int_npy = fix2int(pth_npy, frac_bits)
    with open(target_filename, type) as f:
        for num in int_npy:
            for data in num:
                bin_npy = int2bin(data, total_bits)
                hex_npy = bin2hex(bin_npy, total_bits//4)
                if num_count == num_per_line-1:
                    f.write(hex_npy + '\n')
                    num_count = 0
                else:
                    f.write(hex_npy)
                    num_count += 1   


def trans_W_to_bitW(W, **kwargs):
    BIT_BLOCK = kwargs["BIT_BLOCK"]
    WIDTH_BLOCK = kwargs["WIDTH_BLOCK"] 
    HEIGHT_BLOCK = kwargs["HEIGHT_BLOCK"]

    W_1bit = np.zeros((args.array_size*HEIGHT_BLOCK*WIDTH_BLOCK*BIT_BLOCK, args.array_size)).astype(np.int8)
    for j in range(WIDTH_BLOCK):
        for i in range(HEIGHT_BLOCK):
            W_block = W[i*args.array_size:(i+1)*args.array_size, j*args.array_size:(j+1)*args.array_size]
            Binary_W_8bit = np.unpackbits(W_block.astype(np.uint8).reshape(W_block.shape + (1,)), axis=-1)
            for bit in range(BIT_BLOCK):
                W_1bit[((i+j*HEIGHT_BLOCK)*BIT_BLOCK+bit) * args.array_size : ((i+j*HEIGHT_BLOCK)*BIT_BLOCK+bit+1) * args.array_size, :] = \
                Binary_W_8bit[:,:,8-bit-1].T

    W_1bit_r = W_1bit.reshape(-1,4)     
    W_1bit_compress = np.zeros(W_1bit_r.shape[0]).astype(np.uint8)
    for k in range(W_1bit_r.shape[0]):
        W_1bit_compress[k] = W_1bit_r[k,0] * 8 + W_1bit_r[k,1] * 4 + W_1bit_r[k,2] * 2 + W_1bit_r[k,3]      
    return W_1bit_compress

def little_head(data, saved_line, type="xz"):
    data = data.reshape(saved_line, -1)
    # 创造出来的x_zp, w_zp, fpw, fpb的每一行是从左到右排列，
    # 现在需要按块来从右往左排列。
    # 每块的大小是64
    # 比如：xz是8bit，每行能存512个数。每64个数组成一个块，那么每行是8个块。
    # 本来这8个块的存储顺序为0,1,2,3,4,5,6,7,
    # 现在需要变成：7,6,5,4,3,2,1,0.
    # saved_line: 数据在weight_bram中存储的行数
    if type == "xz":
        num_chunks = int(args.weight_bram_width / args.array_size / args.x_bit)
    elif type == "wz":
        num_chunks = int(args.weight_bram_width / args.array_size / args.wz_bit)
    elif type == "fp":
        num_chunks = int(args.weight_bram_width / args.array_size / 16)
    else:
        raise NotImplementedError("type not supported!")
    for line in range(saved_line):
        chunks = np.split(data[line], num_chunks)
        rearrange_order = list(range(num_chunks-1, -1, -1))
        rearranged_chunks = [chunks[i] for i in rearrange_order]
        rearranged_array = np.concatenate(rearranged_chunks)
        data[line] = rearranged_array
    return data

def reorder_fp_data(FPW, FPB):
    # 这个文件可以将FPW、FPB分组存储。
    # 现在的FPW和FPB存储方式是：先存FPW，后存FPB。
    # 但是，当它们的长度大于1024时，这样存起来在buffer中放不下。
    # 所以要改成:1024个FPW，1024个FPB，然后1024个FPW，1024个FPB，依次类推。
    # 1024这个数据非常友好。因为大模型的中间一些维度很多都是1024的倍数。
    max_block_len = 1024
    fp_len = len(FPW)
    if fp_len <= max_block_len:
        reorderd_fp = torch.cat((FPW, FPB), dim=-1)
        return reorderd_fp
    else:
        reorderd_fp = torch.tensor([])
        num_block = math.ceil(fp_len/max_block_len)
        for i in range(num_block):
            last_index = min((i+1)*max_block_len, fp_len)
            reorderd_fp = torch.cat((reorderd_fp, FPW[i*max_block_len:last_index]), dim=-1)
            reorderd_fp = torch.cat((reorderd_fp, FPB[i*max_block_len:last_index]), dim=-1)
    return reorderd_fp


def create_nonlinear_data():
    # 6个非线性常数，占了112bit
    # 左边的bit都补零     
    QM = 16
    Qlog_d = 85         # 16'b0000_0000_0101_0101
    QM_hex = data2hex(QM, 16, 0)
    Qlog_d_hex = data2hex(Qlog_d, 16, 0)

    # 上面一共有112bit了，现在给4096bit的左边全部补0
    len_const = 7
    total_length = int(args.weight_bram_width / 16)
    zeros = torch.zeros(size=(total_length-len_const,), dtype=torch.int16)
    npy2hex(zeros, data_dir+"saved_data/nonlinear_data.txt", 16, 0, num_per_line=total_length, type='w')
    with open(data_dir+"saved_data/nonlinear_data.txt", 'a') as f:
        f.write(Qlog_d_hex)
        f.write('0000')
        f.write('00000000')
        f.write('0000')
        f.write(Qlog_d_hex)
        f.write(QM_hex)
        f.write("\n")

def create_layernorm_xin():
    # 准备输入给Layernorm层的X，都是8bit数据
    X_filename = f"{pth_prefix}final_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    X = torch.load(X_filename, map_location=torch.device('cpu')).detach()
    npy2hex2D(torch.round(X).numpy(), data_dir+"saved_data/Layernorm_Xin.txt", args.x_bit, 0, args.array_size, "w")         
    return 

def create_fc1_fp_data():
    # Layernorm之后的FP数据
    # 现有的数据是FPW和FPB，(都是FP数据，768个)
    # 但是我们要将它们改成：FPW/s, (FPB/s + Zp - 128)
    NUM_PER_LINE = args.weight_bram_width // 16

    ln_wd2m_source_filename = f"{pth_prefix}final_layer_norm/w_div_2M.pth"
    ln_wd2m = torch.load(ln_wd2m_source_filename, map_location=torch.device('cpu')).detach()

    ln_bias_filename = f"{pth_prefix}final_layer_norm/bias.pth"
    ln_bias = torch.load(ln_bias_filename, map_location=torch.device('cpu')).detach()   

    scale_filename = f"{pth_prefix}final_layer_norm.out_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zero_point_filename = f"{pth_prefix}final_layer_norm.out_quantizer/zeropoint.pth"
    zero_point = torch.load(zero_point_filename, map_location=torch.device('cpu')).detach()

    ln_wd2m_fusion = ln_wd2m / scale
    ln_bias_fusion = ln_bias / scale + zero_point - args.x_shift

    # reorder
    reorderd_fp_data = reorder_fp_data(ln_wd2m_fusion, ln_bias_fusion).numpy()

    # 注意：对每个fp16的数组，存储方式是小端序，也就是低字节存在前面
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/Fc1_FP.txt", num_per_line=NUM_PER_LINE, type="w")
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // NUM_PER_LINE)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/Fc1_FP_v2.txt", num_per_line=NUM_PER_LINE, type="w")
    return 

def create_fc1_weight():
    # FC1层的权重
    # 都是2bit无符号数
    # Bram中每一行存一个64*64*1bit的块
    # 一定要记得转置！
    NUM_PER_LINE = args.weight_bram_width // 4
    HEIGHT_BLOCK = args.hidden_size // args.array_size
    WIDTH_BLOCK = args.ffn_size // args.array_size
    W_filename = f"{pth_prefix}fc1.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()
    W = torch.round(W.T).numpy()
    W_1bit_compress = trans_W_to_bitW(W, 
                                      HEIGHT_BLOCK=HEIGHT_BLOCK,
                                      WIDTH_BLOCK=WIDTH_BLOCK,
                                      BIT_BLOCK=args.weight_bit_block)
    #存16进制的数
    npy2hex(W_1bit_compress, data_dir+"saved_data/Fc1_Weight.txt", total_bits=4, frac_bits=0, num_per_line=NUM_PER_LINE)
    return

def create_fc1_linear():
    # Fc1的-Wz, +Xz-128.
    # 可能需要在末尾补0以凑成一整行。
    # 先存x_zp, 再存w_zp。
    # num_per_line: 每行存多少个数
    # num_per_block: 每行存多少个block, 一个block是64个数
    NUM_PER_LINE1 = args.weight_bram_width // args.x_bit
    NUM_PER_LINE2 = args.weight_bram_width // args.wz_bit
    BLOCK_PER_LINE1 = args.weight_bram_width // (args.x_bit * args.array_size)
    BLOCK_PER_LINE2 = args.weight_bram_width // (args.wz_bit * args.array_size)
    HEIGHT_BLOCK = args.hidden_size // args.array_size
    WIDTH_BLOCK = args.ffn_size // args.array_size

    x_zp_source_filename = f"{pth_prefix}final_layer_norm.out_quantizer/zeropoint.pth"
    x_zp = torch.load(x_zp_source_filename, map_location=torch.device('cpu')).detach()
    x_zp = x_zp.reshape(-1,args.array_size)    
    x_zp = x_zp - args.x_shift
    zero_block = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1) * BLOCK_PER_LINE1 - HEIGHT_BLOCK
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    x_zp = torch.cat((x_zp, zeros), dim=0)
    x_zp = torch.round(x_zp).numpy()

    w_zp_source_filename = f"{pth_prefix}fc1.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()
    w_zp = w_zp.reshape(-1,args.array_size)
    zero_block = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2) * BLOCK_PER_LINE2 - WIDTH_BLOCK
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    w_zp = torch.cat((w_zp, zeros), dim=0)
    w_zp = torch.round(w_zp).numpy()

    npy2hex2D(x_zp, data_dir + "saved_data/Fc1_linear.txt",total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir + "saved_data/Fc1_linear.txt",total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")

    # 开始小头存储,也就是说将每一行中排在左边的x_zp块放在右边
    # 每个块是64个数。
    x_zp_saved_line = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1)
    x_zp = little_head(x_zp, x_zp_saved_line, type="xz")
    w_zp_saved_line = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2)
    w_zp = little_head(w_zp, w_zp_saved_line, type="wz")
    npy2hex2D(x_zp, data_dir + "saved_data/Fc1_linear_v2.txt",total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir + "saved_data/Fc1_linear_v2.txt",total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")


def create_fc2_fp_data():
    # FC1的输出是int32，要继续经过一个FP单元才能变成int8
    # 它需要首先乘以 Sx1*Sw1(Sx1是fc1的输入的scale)
    # 然后使用fc2的输入的(Sx2和zeropoint2)进行量化，变成int8
    # 考虑到权重还有bias部分，所以要将这一部分也融合进zero_point.
    # 所以这个FP的输入的FPW和FPB分别是：Sx1*Sw1/Sx2, （zeropoint2-128）+B/Sx2.
    # 得到都是3072大小的浮点数，都是fp16.
    # 每行存256个数，一共存3072/256*2=24行数据。
    # 不需要补零
    NUM_PER_LINE = args.weight_bram_width // 16
    Sx1_filename = f"{pth_prefix}final_layer_norm.out_quantizer/scale.pth"
    Sx1 = torch.load(Sx1_filename, map_location=torch.device('cpu')).detach()

    Sw1_filename = f"{pth_prefix}fc1.weight_quantizer/scale.pth"
    Sw1 = torch.load(Sw1_filename, map_location=torch.device('cpu')).detach()     

    Sx2_filename = f"{pth_prefix}fc2.act_quantizer/scale.pth"
    Sx2 = torch.load(Sx2_filename, map_location=torch.device('cpu')).detach()

    zp2_filename = f"{pth_prefix}fc2.act_quantizer/zeropoint.pth"
    zp2 = torch.load(zp2_filename, map_location=torch.device('cpu')).detach()       

    bias_filename = f"{pth_prefix}fc1/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    FPW = Sx1 * Sw1 / Sx2                                                          
    FPB = torch.round(zp2) + bias / Sx2 - args.x_shift   
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/Fc2_FP.txt", num_per_line=NUM_PER_LINE, type="w")
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // NUM_PER_LINE)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/Fc2_FP_v2.txt", num_per_line=NUM_PER_LINE, type="w")
    return

def create_fc2_weight():
    # FC2层的权重
    # 都是2bit无符号数
    # Bram中每一行存一个64*64*1bit的块
    # fc1: dequant.pth: 量化权重, scale.pth: 缩放因子, zero_point.pth: 权重量化零点
    # 一共可以存48*12*2=1152行
    NUM_PER_LINE = args.weight_bram_width // 4
    HEIGHT_BLOCK = args.ffn_size // args.array_size
    WIDTH_BLOCK = args.hidden_size // args.array_size
    W_filename = f"{pth_prefix}fc2.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()
    W = torch.round(W.T).numpy()
    W_1bit_compress = trans_W_to_bitW(W,
                                    BIT_BLOCK=args.weight_bit_block,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    HEIGHT_BLOCK=HEIGHT_BLOCK)
    npy2hex(W_1bit_compress, data_dir+"saved_data/Fc2_Weight.txt", total_bits=4, frac_bits=0, num_per_line=NUM_PER_LINE)

def create_fc2_linear():
    # 其中Wz是2bit有符号数据，共768个.但我要把它们当成8bit无符号数。
    # Xz是8bit有符号数据，共3072个
    # 它们都被存为4096bit一行。先存x_zp，后存w_zp.
    WIDTH_BLOCK = args.hidden_size // args.array_size
    NUM_PER_LINE1 = args.weight_bram_width // args.x_bit
    NUM_PER_LINE2 = args.weight_bram_width // args.wz_bit
    BLOCK_PER_LINE1 = args.weight_bram_width // (args.x_bit * args.array_size)
    BLOCK_PER_LINE2 = args.weight_bram_width // (args.wz_bit * args.array_size)
    HEIGHT_BLOCK = args.ffn_size // args.array_size
    WIDTH_BLOCK = args.hidden_size // args.array_size

    x_zp_source_filename = f"{pth_prefix}fc2.act_quantizer/zeropoint.pth"
    x_zp = torch.load(x_zp_source_filename, map_location=torch.device('cpu')).detach()     
    x_zp = x_zp.reshape(-1, args.array_size)
    x_zp = x_zp - args.x_shift
    zero_block = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1) * BLOCK_PER_LINE1 - HEIGHT_BLOCK      
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    x_zp = torch.cat((x_zp, zeros), dim=0)
    x_zp = torch.round(x_zp).numpy()

    w_zp_source_filename = f"{pth_prefix}fc2.weight_quantizer/zeropoint.pth"
    w_zp = torch.load(w_zp_source_filename, map_location=torch.device('cpu')).detach()
    w_zp = w_zp.reshape(-1, args.array_size)
    zero_block = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2) * BLOCK_PER_LINE2 - WIDTH_BLOCK 
    zeros = torch.zeros(size=(zero_block, args.array_size), dtype=torch.int8)
    w_zp = torch.cat((w_zp, zeros), dim=0)
    w_zp = torch.round(w_zp).numpy()

    npy2hex2D(x_zp, data_dir + "saved_data/Fc2_linear.txt",total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir + "saved_data/Fc2_linear.txt",total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")

    # 小头存储
    x_zp_saved_line = math.ceil(HEIGHT_BLOCK / BLOCK_PER_LINE1)
    x_zp = little_head(x_zp, x_zp_saved_line, type="xz")
    w_zp_saved_line = math.ceil(WIDTH_BLOCK / BLOCK_PER_LINE2)
    w_zp = little_head(w_zp, w_zp_saved_line, type="wz")
    npy2hex2D(x_zp, data_dir + "saved_data/Fc2_linear_v2.txt",total_bits=args.x_bit, frac_bits=0, num_per_line=NUM_PER_LINE1, type="w")
    npy2hex2D(w_zp, data_dir + "saved_data/Fc2_linear_v2.txt",total_bits=args.wz_bit, frac_bits=0, num_per_line=NUM_PER_LINE2, type="a")


def create_skip_s():
    # 在进行最后的残差连接时，layer_norm输入的int8需要乘以它的scale，
    # 这个scale就是skip_s. 只是一个数，1维。
    # 考虑到要融合下一层的Attention block的layernorm的in_quantizer, 所以它还要再除以s_next_block.
    s_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    s = torch.load(s_filename, map_location=torch.device('cpu')).item()

    s_next_block_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    s_next_block = torch.load(s_next_block_filename, map_location=torch.device('cpu')).item()

    skip = np.array(s / s_next_block).astype(np.float16)
    bin = fp16_to_binary(skip)
    s_hex = bin2hex(bin, 4)

    len_const = 8
    total_length = int(args.weight_bram_width // 16)
    zeros = torch.zeros(size=(total_length-len_const,), dtype=torch.int16)
    npy2hex(zeros, data_dir+"saved_data/skip_data.txt", 16, 0, num_per_line=total_length, type='w')
    with open(data_dir+"saved_data/skip_data.txt", 'a') as f:
        f.write(s_hex)
        for i in range(0,7):
            f.write("0000")
        f.write("\n")

def create_out_fp_data():
    # FC2的输出是int32，要继续经过一个FP单元才能变成int8
    # 然后才能给下一个block的attention部分用
    # 它需要首先乘以 Sx2*Sw2(Sx2是fc2的输入的scale)
    # 然后使用fc2的输出的(Sx22和zeropoint22)进行量化，变成int8
    # 考虑到权重还有bias部分，所以要将这一部分也融合进zero_point.
    # 将FC的输出量化成8bit之后，会和残差部分相加，成为int9.
    # 然后乘以一个共同的out_scale，变成浮点数。
    # 最后量化成下一个block的输入
    # 所以这个FP的输入的FPW和FPB分别是：（见代码）。
    # 得到都是768大小的浮点数，都是fp16.
    # 每行存256个数，一共存768/256*2=6行数据。
    NUM_PER_LINE = args.weight_bram_width // 16

    Sx_in_filename = f"{pth_prefix}fc2.act_quantizer/scale.pth"
    Sx_in = torch.load(Sx_in_filename, map_location=torch.device('cpu')).detach()

    Sw_filename = f"{pth_prefix}fc2.weight_quantizer/scale.pth"
    Sw = torch.load(Sw_filename, map_location=torch.device('cpu')).detach()  

    Sx_out_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    Sx_out = torch.load(Sx_out_filename, map_location=torch.device('cpu')).detach()

    Zp_out_filename = f"{pth_prefix}final_layer_norm.in_quantizer/zeropoint.pth"
    Zp_out = torch.load(Zp_out_filename, map_location=torch.device('cpu')).detach() # ==0

    bias_filename = f"{pth_prefix}fc2/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    Sx_next_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    Sx_next = torch.load(Sx_next_filename, map_location=torch.device('cpu')).detach()

    FPW = Sx_in * Sw / Sx_next
    FPB = bias / Sx_next + Sx_out * Zp_out / Sx_next
    reorderd_fp_data = reorder_fp_data(FPW, FPB).numpy()
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/out_FP.txt", num_per_line=NUM_PER_LINE, type="w")
    
    # 小头存储
    fp_saved_line = int(len(reorderd_fp_data) // NUM_PER_LINE)
    reorderd_fp_data = little_head(reorderd_fp_data, fp_saved_line, type="fp")
    npy2hex_fp16(np.float16(reorderd_fp_data),data_dir+"saved_data/out_FP_v2.txt", num_per_line=NUM_PER_LINE, type="w")
    return 

def trans_txt_to_mem():
    with open(data_dir+"saved_data/Layernorm_Xin.txt", 'r') as txt:
        with open(data_dir+"saved_data/Layernorm_Xin.mem", 'w') as mem:
            address = 0
            for line in txt:
                mem.write(f'@{address:08X} {line}')
                address += 1

    #file_list = ["nonlinear_data.txt", "Fc1_FP.txt", "Fc1_Weight.txt", "Fc1_linear.txt", "Fc2_FP.txt", "Fc2_Weight.txt", "Fc2_linear.txt", "skip_data.txt", "out_FP.txt"]
    file_list = ["nonlinear_data.txt", "Fc1_FP_v2.txt", "Fc1_Weight.txt", "Fc1_linear_v2.txt", "Fc2_FP_v2.txt", "Fc2_Weight.txt", "Fc2_linear_v2.txt", "skip_data.txt", "out_FP_v2.txt"]
    # 对应的行数分别是：1, 6, 1152, 8, 24, 1152, 8, 1, 6. 一共2358行。
    address = 0
    with open(data_dir+"saved_data/Bram_data.mem", 'w') as mem:
        for file in file_list:
            with open(data_dir+"saved_data/"+file, 'r') as txt:
                for line in txt:
                    mem.write(f'@{address:08X} {line}')
                    address += 1

    with open(data_dir + "saved_data/Bram_data.txt","w") as f:
        for file in file_list:
            with open(data_dir+"saved_data/"+file, 'r') as txt:
                for line in txt:
                    f.write(line)

    # 对于一个bram来说, 如果它有两个端口, 一个宽度是4096， 一个
    # 宽度是1024, 那么默认将初始化的mem文件按照短口写入。即使mem文件有4096列，那也只会截取低位的
    # 1024列写入。所以我现在要做的就是将4096列*2354行的Bram_data.txt文件里面的数据转为1024列*9416行
    # 的数据。对于原来的每一行，先存低位。
    # 另外也要存个txt文件。
    address = 0
    with open(data_dir+"saved_data/Long_bram_data.mem", "w") as mem, \
        open (data_dir+"saved_data/Long_bram_data.txt", "w") as txt:
        with open(data_dir+"saved_data/Bram_data.txt", "r") as f:
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

def llm_create_fc_data():
    # 1. Nonlinear data: for Layernorm
    create_nonlinear_data()

    # 2. Layernorm input: X (2048, 768)
    create_layernorm_xin()

    # 3. Layernorm之后的FP数据(FC1之前)
    create_fc1_fp_data()
    
    # 4. Fc1 weight: ()
    create_fc1_weight()

    # 5. Fc1的-Wz, -Xz
    create_fc1_linear()

    # 6. Fc2之前的FP数据
    create_fc2_fp_data()

    # 7. Fc2 weight: 
    create_fc2_weight()

    # 8. Fc2的-Wz, -Xz
    create_fc2_linear()

    # 9. 遥远的skip_s
    create_skip_s()

    # 10. Fc2的输出的FP数据
    create_out_fp_data()

    # 10. 将上面生成的8个txt文件弄成两个mem文件。
    # 第一个mem文件只存Layernorm的输入X_int
    # 第二个mem文件存其他所有的数据
    trans_txt_to_mem()
    return 

if __name__ == '__main__':
    llm_create_fc_data()