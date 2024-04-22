# transfer data to vivado readable simulation data format
import ctypes
import numpy as np 
import torch 

NONL_LENGTH = 64
MIN_INT8 = -128
MAX_INT8 = 127
MIN_INT16 = -32768
MAX_INT16 = 32767
MIN_INT32 = -2147483648
MAX_INT32 = 2147483647

# def int2bin(x, total_bits): # negative integer should be transferred to 2's complement
#     if x < 0:
#         x = 2**total_bits + x
#     x = int(x)
#     y = bin(x)[2:] 
#     return pretty_bin(y.zfill(total_bits))

def int2bin(n:int, datalen:int) -> str:
    """整数到补码"""
    n = int(n)
    if n >= 0:
        b = format(n, 'b').zfill(datalen)
        length = len(b)
        # 计算需要补齐的位数
        padding = datalen - length
        b = ('0' * padding + b)
    else:
        b = format(2**datalen + n, 'b')
        length = len(b)
        # 计算需要补齐的位数
        padding = datalen - length
        b = ('1' * padding + b)
    return pretty_bin(b)
    # return b

def bin2int(b:str, datalen:int) -> int:
    """补码到整数"""
    if b[0] == '1':
        return -1 * (2**datalen - int(b, 2))
    else:
        return int(b, 2)

def int2hex(n:int, datalen:int) -> str:
    """整数到16进制, datalen为二进制比特数"""
    b = int2bin(n, datalen)
    h = bin2hex(b, datalen//4)
    return h

def hex2int(h:str, datalen:int) -> int:
    """整数到16进制, datalen为二进制比特数"""
    b = hex2bin(h, datalen)
    n = bin2int(b, datalen)
    return n

def bin2hex(b:str, datalen:int) -> str:
    """二进制到16进制"""
    h = hex(int(b, 2))[2:]
    length = len(h)
    # 计算需要补齐的位数
    padding = datalen - length
    
    if b[0] == '0':
        return ('0' * padding + h)  
    else:
        return ('f' * padding + h)

def hex2bin(h:str, datalen:int) -> str:
    """16进制到二进制，补齐4的整数倍位数"""
    b = format(int(h, 16), 'b')
    length = len(b)
    # 计算需要补齐的位数
    padding = datalen - length
    # 添加符号位
    b = ('0' * padding + b) if h >= '0' else ('1' * padding + b)
    return pretty_bin(b)

def pretty_bin(b:str) -> str:
    """美化二进制输出，每四位插入一个下划线"""
    b = b[::-1]
    pretty_b = '_'.join(b[i:i+4] for i in range(0, len(b), 4))
    return pretty_b[::-1]

def fix2int(x, total_bits, frac_bits):
    y = np.round(x*2**frac_bits)
    return y

def pth2npy(source_filename):
    pth = torch.load(source_filename, map_location=torch.device('cpu'))
    pth_npy = pth.numpy()
    return pth_npy

def pth2bin(source_filename, target_filename, total_bits, frac_bits, num_per_line=1):
    """将pth文件转换为Vivado仿真可读取的bin文件"""
    num_count = 0
    pth_npy = pth2npy(source_filename).reshape(-1)
    int_npy = fix2int(pth_npy, total_bits, frac_bits)
    with open(target_filename, 'w') as f:
        for num in int_npy:
            bin_npy = int2bin(num, total_bits)
            if num_count == num_per_line-1:
                f.write(bin_npy + '\n')
                num_count = 0
            else:
                f.write(bin_npy)
                num_count += 1

def npy2bin(pth_npy, target_filename, total_bits, frac_bits, num_per_line=1):
    """将numpy数组转换为Vivado仿真可读取的bin文件"""
    num_count = 0
    int_npy = fix2int(pth_npy, total_bits, frac_bits)
    with open(target_filename, 'w') as f:
        for num in int_npy:
            bin_npy = int2bin(num, total_bits)
            if num_count == num_per_line-1:
                f.write(bin_npy + '\n')
                num_count = 0
            else:
                f.write(bin_npy)
                num_count += 1                 

def pth2hex(source_filename, target_filename, total_bits, frac_bits, num_per_line=1):
    """将pth文件转换为Vivado仿真可读取的hex文件"""
    num_count = 0
    pth_npy = pth2npy(source_filename).reshape(-1) # transfer numpy array to vector
    int_npy = fix2int(pth_npy, total_bits, frac_bits)
    with open(target_filename, 'w') as f:
        for num in int_npy:
            bin_npy = int2bin(num, total_bits)
            hex_npy = bin2hex(bin_npy, total_bits//4)
            if num_count == num_per_line-1:
                f.write(hex_npy + '\n')
                num_count = 0
            else:
                f.write(hex_npy)
                num_count += 1  

def npy2hex(pth_npy, target_filename, total_bits, frac_bits, num_per_line=1):
    """将numpy数组转换为Vivado仿真可读取的hex文件"""
    num_count = 0
    int_npy = fix2int(pth_npy, total_bits, frac_bits)
    with open(target_filename, 'w') as f:
        for num in int_npy:
            bin_npy = int2bin(num, total_bits)
            hex_npy = bin2hex(bin_npy, total_bits//4)
            if num_count == num_per_line-1:
                f.write(hex_npy + '\n')
                num_count = 0
            else:
                f.write(hex_npy)
                num_count += 1     

def fp16_to_binary(half_num):
    binary_str = format(int.from_bytes(half_num.tobytes(), byteorder='little'), '016b')
    return binary_str
 
def npy2hex_fp16(pth_npy, target_filename, num_per_line=1, type="w"):
    """将FP16数据转换为Vivado仿真可读取的hex文件"""
    num_count = 0 
    pth_npy = pth_npy.reshape(-1)
    with open(target_filename, type) as f:
        for num in pth_npy:
            bin_npy = fp16_to_binary(num)
            hex_npy = bin2hex(bin_npy, 4)
            if num_count == num_per_line-1:
                f.write(hex_npy + '\n')
                num_count = 0
            else:
                f.write(hex_npy)
                num_count += 1

def sum4hex(hexstr, stride=2, datalen=8):
    sum = 0
    for i in range(0,len(hexstr),stride):
        h = hex2int(hexstr[i:i+stride],datalen)
        sum += h
    return sum

def temp_check1():
    '''用于检查移位与四舍五入的差别'''
    print("Negative values", int2hex(-4144*85,32), int2bin(-4144*85,32), bin2int('1111_1111_1111_1010', 16), -4144/768)
    print("Negative values", int2hex(-1144*85,32), int2bin(-1144*85,32), bin2int('1111_1111_1111_1110', 16), -1144/768)
    print("Positive values", int2hex(4144*85,32), int2bin(4144*85,32), bin2int('0000_0000_0000_0101', 16), 4144/768)
    print("Positive values", int2hex(1544*85,32), int2bin(1544*85,32), bin2int('0000_0000_0000_0010', 16), 1544/768)
    print("Negative values", int2hex(-304*85,32), int2bin(-304*85,32), bin2int('1111_1111_1111_1111', 16), -304/768)

def main():
    aug_check = False
    np.random.seed(0)
    ''' 临时检验结果 '''
    '''
    1/768 ~ 1/1024+1/4096+1/16384+1/65536  #0000_0000_0101_0101
    1/2560 ~ 1/4096+1/8192+1/65536
    '''

    ''' 仿真数据验证：通用模块'''
    data_dir = "vivado_sim/"
    # # 向量减法
    # Feature = np.random.randint(-128, 127, (128))
    # Subtrahend = np.max(Feature)
    # Subtrahend = np.array([Subtrahend])
    # Res = Feature - Subtrahend
    # npy2hex(Feature, data_dir+"minus_feature.txt", total_bits=8, frac_bits=0, num_per_line=64)
    # npy2hex(Subtrahend, data_dir+"minus_subtrahend.txt", total_bits=8, frac_bits=0, num_per_line=1)
    # npy2hex(Res, data_dir+"minus_res.txt", total_bits=16, frac_bits=0, num_per_line=64)
    # print(Feature[0:10])
    # print(Res[0:10])

    # # 向量加法
    # Feature = np.random.randint(-128, 127, (128))
    # Addend = np.random.randint(-128, 127, (128))
    # Addend2 = np.random.randint(-128, 127, (1))
    # Res = Feature + Addend + Addend2
    # npy2hex(Feature, data_dir+"add_feature.txt", total_bits=32, frac_bits=0, num_per_line=64)
    # npy2hex(Addend, data_dir+"add_addend.txt", total_bits=16, frac_bits=0, num_per_line=64)
    # npy2hex(Addend2, data_dir+"add_addend2.txt", total_bits=16, frac_bits=0, num_per_line=1)
    # npy2hex(Res, data_dir+"add_res.txt", total_bits=32, frac_bits=0, num_per_line=64)
    # print(Feature[0:10])
    # print(Addend[0:10])
    # print(Addend2[0])
    # print(Res[0:10])

    # # 向量乘法
    # Multiplier1 = np.random.randint( -128, 127, (128) )
    # Multiplier2 = np.random.randint( -128, 127, (128) )
    # Res = Multiplier1 * Multiplier2
    # npy2hex(Multiplier1, data_dir+"mul_multiplier1.txt", total_bits=16, frac_bits=0, num_per_line=64)
    # npy2hex(Multiplier2, data_dir+"mul_multiplier2.txt", total_bits=16, frac_bits=0, num_per_line=64)
    # npy2hex(Res, data_dir+"mul_res.txt", total_bits=32, frac_bits=0, num_per_line=64)

    # 比较树
    # Feature = np.random.randint(-128, 127, (64))
    # Res = np.array([np.max(Feature)])
    # npy2hex(Feature, data_dir+"cmp_feature.txt", total_bits=8, frac_bits=0, num_per_line=64)
    # npy2hex(Res, data_dir+"cmp_res.txt", total_bits=8, frac_bits=0, num_per_line=1)

    # # 加法树
    # Feature = np.random.randint(-128, 127, (64))
    # Res = np.array([np.sum(Feature)])
    # npy2hex(Feature, data_dir+"addtree_feature.txt", total_bits=8, frac_bits=0, num_per_line=64)
    # npy2hex(Res, data_dir+"addtree_res.txt", total_bits=16, frac_bits=0, num_per_line=1)

    '''真实数据验证： opt-125M'''
    # ##softmax stage1
    xint_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/x_int.pth"
    xint_target_filename = "vivado/xint.txt"
    pth_int = torch.load(xint_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_int[0, -1, :].numpy(), xint_target_filename, total_bits=8, frac_bits=0, num_per_line=64)
    # print("Input x_int shape", pth_int.shape, pth_int[0,-1,0:64], pth_int[0,-1,64:128])     

    xmax_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/x_int_max.pth"
    xmax_target_filename = "vivado/xmax.txt"
    pth_max = torch.load(xmax_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_max[0,-1,:].numpy(), xmax_target_filename, total_bits=8, frac_bits=0, num_per_line=1)
    # print("Input x_max shape", pth_max.shape, pth_max[0,-1,:], torch.max(pth_int[0,-1,0:64]))  

    xfix_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/x_int_sub_max.pth"
    xfix_target_filename = "vivado/xfix.txt"
    pth_fix = torch.load(xfix_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_fix[0, -1, :].numpy(), xfix_target_filename, total_bits=16, frac_bits=0, num_per_line=64)
    print("xfix", pth_fix.shape, pth_fix[0,-1,0]) 

    # ##softmax stage2 
    t1_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/t1.pth"  #可能并不是一个fix16的值
    t1_target_filename = "vivado/t1_Qs_div_ln2.txt"
    pth_t1 = torch.load(t1_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_t1.numpy(), t1_target_filename, total_bits=16, frac_bits=16)
    print("t1", pth_t1.numpy(), pth_t1*(2**32), int2hex(pth_t1*(2**32),32)) 

    t2_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/t2.pth" 
    t2_target_filename = "vivado/t2_Qln2_div_s.txt" 
    pth_t2 = torch.load(t2_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_t2.numpy(), t2_target_filename, total_bits=16, frac_bits=8)
    print("t2", pth_t2.numpy(), pth_t2*(2**8), int2hex(pth_t2*(2**8),16))

    t3_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/t3.pth" 
    t3_target_filename = "vivado/t3_Q2_div_s.txt"
    pth_t3 = torch.load(t3_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_t3.numpy(), t3_target_filename, total_bits=32, frac_bits=8)   
    print("t3", pth_t3.numpy(), pth_t3*(2**8), int2hex(pth_t3*(2**8),32))

    z_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/z.pth"
    z_target_filename = "vivado/z.txt"
    pth_z = torch.load(z_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_z[0,-1,:].numpy(), z_target_filename, total_bits=16, frac_bits=0, num_per_line=64)
    print("z", pth_z[0,-1,0], pth_z[0,-1,0], int2hex(pth_z[0,-1,0],16))

    p_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/p.pth"
    p_target_filename = "vivado/p.txt"
    pth_p = torch.load(p_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_p[0,-1,:].numpy(), p_target_filename, total_bits=32, frac_bits=8, num_per_line=64)
    print("p", pth_p[0,-1,0], pth_p[0,-1,0]*(2**8), int2hex(pth_p[0,-1,0]*(2**8),32))

    v_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/v.pth"
    v_target_filename = "vivado/v.txt"
    pth_v = torch.load(v_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_v[0,-1,:].numpy(), v_target_filename, total_bits=32, frac_bits=16, num_per_line=64)
    print("v", pth_v[0,-1,0], pth_v[0,-1,0]*(2**16), int2hex(pth_v[0,-1,0]*(2**16),32) )

    print(pth_v.shape)
    v_temp = pth_v[0,-1,:].numpy()*(2**16)
    v_accu = 0
    for i in range(32):
        v_sum_temp = np.sum(v_temp[i*64:(i+1)*64])
        v_accu = v_accu + v_sum_temp
        print("Step", i, int2hex(v_sum_temp,32), int2hex(v_accu,32))

    ## print("Simulating addition:")
    ## x_fix_0 = pth_fix[0,-1,5].numpy()
    ## t1_0 = pth_t1.numpy()  
    ## t2_0 = pth_t2.numpy() 
    ## t3_0 = pth_t3.numpy()
    ## z_0 = pth_z[0,-1,5].numpy()  
    ## p_check = x_fix_0*2**8 + z_0*(t2_0*2**8) + t3_0*2**8
    ## print("p_check", p_check, pth_p[0,-1,5]*2**8)

    vsum_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/v_sum.pth"
    vsum_target_filename = "vivado/v_sum.txt"
    pth_vsum = torch.load(vsum_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_vsum[0,-1,:].numpy(), vsum_target_filename, total_bits=32, frac_bits=16, num_per_line=1)

    ## softmax stage3 
    result_source_filename = "tensors/model.decoder.layers.0.self_attn.softmax/result.pth"
    result_target_filename = "vivado/result.txt"
    pth_result = torch.load(result_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_result[0,-1,:].numpy(), result_target_filename, total_bits=32, frac_bits=16, num_per_line=64)
    # print("I am ok", pth_v[0,-1,0], pth_vsum[0,-1,:], pth_result[0,-1,0])
    # print(pth_v[0,-1,0]*2**16, pth_vsum[0,-1,:]*2**16, pth_result[0,-1,0:2]*2**16)

    ## layernorm stage1 
    ln_xint_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/x_int.pth"
    ln_xint_target_filename = "vivado/ln_xint.txt"
    pth_ln_xint = torch.load(ln_xint_source_filename, map_location=torch.device('cpu'))
    if aug_check:
        pth_ln_xint = pth_ln_xint - 5
    npy2hex(pth_ln_xint[0, -1, :].numpy(), ln_xint_target_filename, total_bits=8, frac_bits=0, num_per_line=64)
    print("Input x_int shape", pth_ln_xint.shape, pth_ln_xint[0,-1,0], torch.sum(pth_ln_xint[0,-1,:]) )

    ln_xmean_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/mean_int.pth"
    ln_xmean_target_filename = "vivado/ln_xmean.txt"
    pth_ln_xmean = torch.load(ln_xmean_source_filename, map_location=torch.device('cpu'))
    if aug_check:
        pth_ln_xmean = pth_ln_xmean - 5
    npy2hex(pth_ln_xmean[0, -1, :].numpy(), ln_xmean_target_filename, total_bits=16, frac_bits=0, num_per_line=1)
    print("Input x_mean shape", pth_ln_xmean.shape, pth_ln_xmean[0,-1,:], torch.mean(pth_ln_xmean[0,-1,:]))

    ln_yint_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/y_int.pth"
    ln_yint_target_filename = "vivado/ln_yint.txt"
    pth_ln_yint = torch.load(ln_yint_source_filename, map_location=torch.device('cpu'))
    if aug_check:
        pth_ln_yint = pth_ln_yint - 5
    npy2hex(pth_ln_yint[0, -1, :].numpy(), ln_yint_target_filename, total_bits=8, frac_bits=0, num_per_line=64)
    print("Input y_int shape", pth_ln_yint.shape, pth_ln_yint[0,-1,0])

    ## layernorm stage2 
    ln_varint_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/var_int.pth"
    ln_varint_target_filename = "vivado/ln_varint.txt"
    pth_ln_varint = torch.load(ln_varint_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_ln_varint[0, -1,:].numpy(), ln_varint_target_filename, total_bits=32, frac_bits=0, num_per_line=1)
    print("Input var_int shape", pth_ln_varint.shape, pth_ln_varint[0,-1,:], torch.sum(pth_ln_yint**2, dim=-1)[:,-1])
    ln_varsum_target_filename = "vivado/ln_varsum.txt"
    npy2hex( torch.sum(pth_ln_yint**2, dim=-1)[:,-1].numpy(), ln_varsum_target_filename, total_bits=32, frac_bits=0, num_per_line=1 )
    print( "Input var_sum value", torch.sum(pth_ln_yint**2, dim=-1)[:,-1] )

    ## layernorm stages
    ln_stdint_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/std_int.pth" 
    ln_stdint_target_filename = "vivado/ln_stdint.txt" 
    pth_ln_stdint = torch.load(ln_stdint_source_filename, map_location=torch.device('cpu'))
    npy2hex(pth_ln_stdint[0, -1,:].numpy(), ln_stdint_target_filename, total_bits=16, frac_bits=0, num_per_line=1)
    print("Input std_int shape", pth_ln_stdint.shape, pth_ln_stdint[0,-1,:])
    
    ## layernorm additional 
    ln_wd2m_source_filename = "tensors/model.decoder.layers.0.self_attn_layer_norm/w_div_2M.pth"
    ln_wd2m_target_filename = "vivado/ln_wd2m.txt"
    pth_ln_wd2m = torch.load(ln_wd2m_source_filename, map_location=torch.device('cpu'))
    # npy2hex_fp16(pth_ln_wd2m[0, -1,:].numpy(), ln_wd2m_target_filename, total_bits=32, frac_bits=0, num_per_line=1)
    # print( fp16_to_binary(  np.float16(pth_ln_wd2m[0].numpy())  ) )
    npy2hex_fp16( np.float16(pth_ln_wd2m.numpy()), ln_wd2m_target_filename, num_per_line=64 )

    # ## fc test data 
    q_proj_w_source_filename = "tensors/model.decoder.layers.0.self_attn.q_proj.weight_quantizer/dequant.pth"
    q_proj_w = torch.load(q_proj_w_source_filename, map_location=torch.device('cpu'))
    print(q_proj_w.shape, q_proj_w[0,0:10])

    q_proj_wb_source_filename = "tensors/model.decoder.layers.0.self_attn.q_proj.weight_quantizer/zeropoint.pth"
    q_proj_wb = torch.load(q_proj_wb_source_filename, map_location=torch.device('cpu'))
    print(q_proj_wb.shape, q_proj_wb[0:16])

    q_proj_s_source_filename = "tensors/model.decoder.layers.0.self_attn.q_proj.weight_quantizer/scale.pth"
    q_proj_s = torch.load(q_proj_s_source_filename, map_location=torch.device('cpu'))
    print(q_proj_s.shape, q_proj_s[0:16])

    # layernorm out_quantizer:
    fc_ln_out_scale_source_filename = "tensors/model.decoder.layers.0.final_layer_norm/scale.pth"
    fc_ln_out_scale = torch.load(fc_ln_out_scale_source_filename, map_location=torch.device('cpu'))
    print("fc ln out scale:", fc_ln_out_scale.shape, fc_ln_out_scale[0])




    ## fc1: dequant.pth: 量化权重, scale.pth: 缩放因子, zero_point.pth: 权重量化零点
    fc1_dequant_source_filename = "tensors/model.decoder.layers.0.fc1.weight_quantizer/dequant.pth"
    fc1_dequant = torch.load(fc1_dequant_source_filename, map_location=torch.device('cpu'))
    print("fc1 dequant:", fc1_dequant.shape, fc1_dequant[0,0:10])

    fc1_scale_source_filename = "tensors/model.decoder.layers.0.fc1.weight_quantizer/scale.pth"
    fc1_scale = torch.load(fc1_scale_source_filename, map_location=torch.device('cpu'))
    print("fc1 scale:", fc1_scale.shape, fc1_scale[0])

    fc1_weight_source_filename = "tensors/model.decoder.layers.0.fc1.weight_quantizer/x_int.pth"
    fc1_weight = torch.load(fc1_weight_source_filename, map_location=torch.device('cpu'))
    print("fc1 x_int:", fc1_weight.shape, fc1_weight[0,0:10], torch.max(fc1_weight[0,:]), torch.min(fc1_weight[0,:]))

    fc1_zp_source_filename = "tensors/model.decoder.layers.0.fc1.weight_quantizer/zeropoint.pth"
    fc1_zp = torch.load(fc1_zp_source_filename, map_location=torch.device('cpu'))
    print("fc1 zero_point:", fc1_zp.shape, fc1_zp[0])

    fc2_act_scale_source_filename = "tensors/model.decoder.layers.0.fc2.act_quantizer/scale.pth"
    fc2_act_scale = torch.load(fc2_act_scale_source_filename, map_location=torch.device('cpu'))
    print("fc1 act scale", fc2_act_scale.shape)







if __name__ == "__main__":
    main() 
    hexstr = "f080370c12171012d81cd100cff7e7e4d6e5fff047c6000207e716c6f204211518e5f282ec0531d30512deb6ed26072517d9fb18e6f1fce803fa40031929f706"
    print(sum4hex(hexstr, stride=2, datalen=8))
    print(int2hex(-368,16))
