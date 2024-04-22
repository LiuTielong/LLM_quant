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

def nbits(x):
    binary_str = bin(x)[2:]
    return len(binary_str)

def I_SQRT(data_in):
    if data_in == 0:
        return 0
    else:
        x0 = nbits(data_in)/2
        x0 = math.ceil(x0)     
        x = math.pow(2, x0)
        for i in range(4):
            # x_temp = (x + round(data_in / x))/2  # 用round其实更精确
            x_temp = (x + math.floor(data_in / x))/2
            x = math.floor(x_temp)
    return x

def trans_W_to_1bit(W, HEIGHT_BLOCK, WIDTH_BLOCK, BIT_BLOCK):
    ARRAY_SIZE = args.array_size
    W_1bit = np.zeros((ARRAY_SIZE*HEIGHT_BLOCK*WIDTH_BLOCK*BIT_BLOCK, ARRAY_SIZE)).astype(np.int8)
    for j in range(WIDTH_BLOCK):
        for i in range(HEIGHT_BLOCK):
            W_block = W[i*ARRAY_SIZE:(i+1)*ARRAY_SIZE, j*ARRAY_SIZE:(j+1)*ARRAY_SIZE]
            Binary_W_8bit = np.unpackbits(W_block.astype(np.uint8).reshape(W_block.shape + (1,)), axis=-1)
            for bit in range(BIT_BLOCK):
                W_1bit[((i+j*HEIGHT_BLOCK)*BIT_BLOCK+bit) * ARRAY_SIZE : ((i+j*HEIGHT_BLOCK)*BIT_BLOCK+bit+1) * ARRAY_SIZE, :] = \
                Binary_W_8bit[:,:,8-bit-1].T
    return W_1bit

def Matmul_compute(W, Wz, X, Xz, **kwargs):
    # 执行矩阵乘法
    # (W - Wz) @ (X - Xz)
    VECTOR_BLOCK = kwargs["VECTOR_BLOCK"]
    HEIGHT_BLOCK = kwargs["HEIGHT_BLOCK"]
    WIDTH_BLOCK = kwargs["WIDTH_BLOCK"]
    BIT_BLOCK = kwargs["BIT_BLOCK"]
    ARRAY_SIZE = args.array_size

    W_1bit = trans_W_to_1bit(W, HEIGHT_BLOCK=HEIGHT_BLOCK, WIDTH_BLOCK=WIDTH_BLOCK, BIT_BLOCK=BIT_BLOCK)
    
    Shift_A = X - Xz
    Wz_spand = np.tile(Wz, (HEIGHT_BLOCK*ARRAY_SIZE, 1))
    Assym_out = np.matmul(Shift_A, Wz_spand)

    Bit_stream_out = np.zeros((VECTOR_BLOCK*WIDTH_BLOCK, ARRAY_SIZE))
    for i in range(VECTOR_BLOCK):
        for j in range(WIDTH_BLOCK):
            for k in range(HEIGHT_BLOCK):
                W_temp = np.zeros((ARRAY_SIZE, ARRAY_SIZE))
                for bit in range(BIT_BLOCK):
                    W_temp += W_1bit[((j*HEIGHT_BLOCK+k)* BIT_BLOCK + bit) * ARRAY_SIZE : ((j*HEIGHT_BLOCK+k)*BIT_BLOCK+bit+1) * ARRAY_SIZE, :].T * 2**bit
                Bit_stream_out[(i*WIDTH_BLOCK+j),:] += np.matmul(Shift_A[i, k*ARRAY_SIZE : (k+1) * ARRAY_SIZE], W_temp)
    Bit_stream_out = Bit_stream_out.reshape(VECTOR_BLOCK, WIDTH_BLOCK*ARRAY_SIZE)
    Matmul_out = Bit_stream_out - Assym_out

    # 直接计算Matmul_out, 和上面结果相等
    # shift_W = W - Wz
    # Matmul_out_v2 = np.matmul(Shift_A, shift_W)
    return Matmul_out

def Decode_txt_to_int(filename, bit=4, sign =True):
    # 读取的文件里面，只能有一种格式。比如数据全都是int4
    # 不能有数据是int4，有数据是int8
    Out = []
    read_width = bit // 4
    if bit == 4:
        and_value = 0x8
        shift = 16
    elif bit == 8:
        and_value = 0x80
        shift = 256
    elif bit == 32:
        and_value = 0x80000000
        shift = 2**32

    with open(filename, "r") as file:
        hex_strings = file.read().replace('\n', '')
    len_hex = len(hex_strings)
    for i in range(0, len_hex, read_width):
        hex_str = hex_strings[i: i+read_width]
        hex_integer = int(hex_str, 16)
        if sign:
            if hex_integer & and_value:
                hex_integer = hex_integer - shift
        Out.append(hex_integer)
    
    Out = np.array(Out)
    return Out

def Decode_txt_to_2int(filename, **Kwargs):
    # 一个txt文件内部包含2部分数据，它们有各自的长度，以及各自对应的bit位宽
    LEN_A = Kwargs["LEN_A"]
    LEN_B = Kwargs["LEN_B"]
    BIT_A = Kwargs["BIT_A"]
    BIT_B = Kwargs["BIT_B"]
    SIGN_A = Kwargs["SIGN_A"]
    SIGN_B = Kwargs["SIGN_B"]
    read_width_a = BIT_A // 4
    read_width_b = BIT_B // 4
    
    if BIT_A == 4:
        and_value_a = 0x8
        shift_a = 16
    elif BIT_A == 8:
        and_value_a = 0x80
        shift_a = 256
    
    if BIT_B == 4:
        and_value_b = 0x8
        shift_b = 16
    elif BIT_B == 8:
        and_value_b = 0x80
        shift_b = 256

    with open(filename, "r") as file:
        hex_strings = file.read().replace('\n', '')
    A = []
    B = []  
    for i in range(0, LEN_A*read_width_a, read_width_a):
        hex_str = hex_strings[i:i+read_width_a]
        hex_integer = int(hex_str, 16)
        if SIGN_A:
            if hex_integer & and_value_a:
                hex_integer = hex_integer - shift_a
        A.append(hex_integer)
    for i in range(LEN_A*read_width_a, LEN_A*read_width_a+LEN_B*read_width_b, read_width_b):
        hex_str = hex_strings[i:i+read_width_b]
        hex_integer = int(hex_str, 16)
        if SIGN_B:
            if hex_integer & and_value_b:
                hex_integer = hex_integer - shift_b
        B.append(hex_integer)
    A = np.array(A)
    B = np.array(B)
    return A, B

def Decode_txt_to_FP16(filename):
    # 将txt文件全部转换为FP16数据
    with open(filename, "r") as file:
        hex_strings = file.read().replace('\n', '')
    FP = []
    for i in range(0, len(hex_strings), 4):
        hex_str = ''.join(hex_strings[i:i+4])
        hex_str = hex_str[2:] + hex_str[:2]                             # 小尾端存储
        FP.append(struct.unpack('e', bytes.fromhex(hex_str))[0])        # 转化为FP16数据, 其实把'e'换成'>e'就不需要上面一行了
    FP = np.array(FP)
    return FP

def de_reorder_fp_data(FP):
    # 在生成FP数据时，使用了一个函数叫作reorder_fp_data()
    # 我们这里要把用Decode_txt_to_FP16()函数读取的FP数据还原回来
    # 已知max_block_len=2048
    max_block_len=2048
    fp_len = len(FP) // 2
    if fp_len <= max_block_len:
        FPW = FP[0:fp_len]
        FPB = FP[fp_len:]
    else:
        FPW = np.array([])
        FPB = np.array([])
        num_block = math.ceil(fp_len / max_block_len)
        last_index = 0
        for i in range(num_block):
            concat_len = min(max_block_len, fp_len - i * max_block_len)
            FPW = np.concatenate((FPW, FP[last_index : last_index+concat_len]))
            FPB = np.concatenate((FPB, FP[last_index+concat_len : last_index+concat_len*2]))
            last_index = last_index + concat_len * 2
    return FPW, FPB

def test_layernorm_out():
    # layernorm的输出: int 32
    # 经过FP之后, 为int 8.
    # 我要测试的就是经过FP模块之后的结果
    # 形状: [2048, 768]
    print("开始测试layernorm的输出。")
    HEIGHT_BLOCK = 12
    NUM_PER_LINE = args.array_size

    # 1. 读取txt文件，获得LN的输入
    LN_Xin = Decode_txt_to_int(filename=data_dir+'saved_data/Layernorm_Xin.txt', bit=8)
    LN_Xin = LN_Xin.reshape(args.vector_block, HEIGHT_BLOCK*args.array_size)

    # 2. 模拟LN运算
    one_div_d = 5 * 1 / 16**3 + 5 * 1 / 16**4                                               # 1 / 768
    M = 16
    miu_fix = np.sum(LN_Xin, axis=1, keepdims=True) * (one_div_d)                           # stage1的结果， [2048]
    
    miu_fix_round = miu_fix.round()                                                         # 将miu_fix转为int16
    npy2hex2D(miu_fix_round, data_dir+"test_data/Layernorm_stage1_out.txt", 16, 0, 1, "w")

    x_fix = LN_Xin - miu_fix_round                                                          # 就是int16                              
    npy2hex2D(np.round(x_fix), data_dir+"test_data/Layernorm_xfix_out.txt", 16,  0, 1, "w")

    S_fix = np.sum(np.square(x_fix), axis=1, keepdims=True)                            

    S_fix_round = np.round(S_fix)                                                           # int32   
    npy2hex(S_fix_round, data_dir+"test_data/Layernorm_stage2_out_S.txt", 32,  0, 1, "w")

    var = np.round(S_fix * one_div_d).astype(int).reshape(-1)                               # int32
    npy2hex(var, data_dir+"test_data/Layernorm_stage2_out_var.txt", 32,  0, 1, "w")

    sigma_fix = np.array([I_SQRT(var[i]) for i in range(var.shape[0])]).reshape(-1,1)       # int16
    npy2hex(sigma_fix, data_dir+"test_data/Layernorm_stages_out.txt", 32,  0, 1, "w")

    x_fn = np.round(2**M * x_fix / sigma_fix)                                               # int32
    # print(x_fn[0,0:100])

    # 真实的layernorm的输出:

    # 3. 最后的浮点数乘法
    FP = Decode_txt_to_FP16(data_dir +'saved_data/Fc1_FP.txt')
    FPW, FPB = de_reorder_fp_data(FP)


    """实验: 如果不将FPW存为fp16之后再读取, 就只有112个数有偏差。但是如果
    转了之后, 有偏差的数据达到24386个。
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
    FPW = ln_wd2m_fusion
    FPB = ln_bias_fusion
    """

    y = np.clip(np.round(FPW * x_fn + FPB), -128, 127)       
    # print(y[0,0:100])

    # 4. 接下来读取FC1的输入
    dequant_filename = f"{pth_prefix}final_layer_norm.out_quantizer/dequant.pth"
    dequant = torch.load(dequant_filename, map_location=torch.device('cpu')).detach()

    scale_filename = f"{pth_prefix}final_layer_norm.out_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zeropoint_filename = f"{pth_prefix}final_layer_norm.out_quantizer/zeropoint.pth"
    zeropoint = torch.load(zeropoint_filename, map_location=torch.device('cpu')).detach()

    x_fc1 = dequant / scale + (zeropoint - 128)

    y = torch.tensor(y)
    different_pos = torch.nonzero(y != x_fc1)
    print("Layernorm的输出不相等的数据的个数:",different_pos.shape[0])

    """
    # 5.结论: 发现两组数据基本上相等, 只有1以内的差别。这是因为 
    # I_sqrt的算法本身带来了+-1的误差。以及我们这里模拟的Layernorm
    # 算法也带来了一些误差, 比如one_div_d就是用16位定点数模拟的。
    # 这本身不会带来太大的精度损失。有24386个数不一致。
    """

    # 6. 数据存储：这里我存layernorm的输出以及第一个FP的输出
    # layernorm的输出：32bit。每行存64个数，共计2048*12=24576行
    npy2hex2D(x_fn, data_dir+"test_data/Layernorm_Xout.txt", 32, 0, NUM_PER_LINE, "w")
    # 第一个FP的输出: 8bit, 每行存64个数，共计2048*12=24576行
    npy2hex2D(y, data_dir+"test_data/FP_quantizer1_out.txt", 8, 0, NUM_PER_LINE, "w")
    return 

def test_fc1_out():
    """
    一个困扰我很久的致命问题: 将tensor转换为numpy之后, 
    再转换回tensor时, 如果要变成整数, 那就应该先round, 而不是
    直接to(torch.int32)。
    在下面的例子中: Wa != Wb
    W = W.numpy()
    Wa = torch.round(torch.tensor(W)).to(torch.int32)
    Wb = torch.tensor(W).to(torch.int32).
    总结起来就一句话: 在numpy和tensor之间进行相互转化的时候, 应该先round一下。
    """
    # fc1的输出：int32
    # 经过第二个FP模块之后的输出：int 8
    # 我们比对FP之后的输出
    # 然后存储FC1的输出和FP之后的输出
    # 形状： [2048, 3072]
    print("开始测试FC1层的输出。")
    HEIGHT_BLOCK = args.hidden_size // args.array_size
    WIDTH_BLOCK = args.ffn_size // args.array_size

    # 1. 读取txt文件，获得fc1的输入W 
    # 懒得把16进制数据一行行转为矩阵了，我就直接用W在存入Fc1_weight之前的数据
    weight_dequant_source_filename = f"{pth_prefix}fc1.weight_quantizer/dequant.pth"
    weight_dequant = torch.load(weight_dequant_source_filename, map_location=torch.device('cpu'))

    weight_scale_source_filename = f"{pth_prefix}fc1.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_source_filename, map_location=torch.device('cpu'))

    weight_zp_source_filename = f"{pth_prefix}fc1.weight_quantizer/zeropoint.pth"
    weight_zp = torch.load(weight_zp_source_filename, map_location=torch.device('cpu'))

    W = (weight_dequant.T / weight_scale + weight_zp).detach()
    W = torch.round(W).numpy()          # 不能直接用W.numpy(), 而是应该先round

    # 2. Wz, Xz
    # 存储格式为: 
    # Xz: 存了2行, 每行512个8bit有符号数, 第二行只有前一半有值，一共768个。
    # Wz: 存了6行，每行512个8bit无符号数, 一共3072个
    len_xz = args.hidden_size                                               # 这是Xz实际的长度。
    len_wz = args.ffn_size
    num_xz_per_line = args.weight_bram_width // args.x_bit
    num_wz_per_line = args.weight_bram_width // args.wz_bit
    len_xz_save = math.ceil(len_xz / num_xz_per_line) * num_xz_per_line     # 这是Xz在bram中存储的数据数，补了一些0
    len_wz_save = math.ceil(len_wz / num_wz_per_line) * num_wz_per_line
    Xz, Wz = Decode_txt_to_2int(
        filename=data_dir+"saved_data/Fc1_linear.txt",
        LEN_A = len_xz_save,
        LEN_B = len_wz_save,
        BIT_A = 8,
        BIT_B = 8,
        SIGN_A = True,
        SIGN_B = False
    )
    Xz = Xz[0 : len_xz]
    Wz = Wz[0 : len_wz]

    # 3. Layernorm后面的FP1输出的激活值
    # 我这里运算还是使用pth文件中的数据
    dequant_filename = f"{pth_prefix}final_layer_norm.out_quantizer/dequant.pth"
    dequant = torch.load(dequant_filename, map_location=torch.device('cpu')).detach()

    scale_filename = f"{pth_prefix}final_layer_norm.out_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zeropoint_filename = f"{pth_prefix}final_layer_norm.out_quantizer/zeropoint.pth"
    zeropoint = torch.load(zeropoint_filename, map_location=torch.device('cpu')).detach()

    Xin = (torch.clamp(torch.round(dequant / scale + zeropoint-128), -128, 127))
    Xin = Xin.detach().numpy()

    # 4.进行矩阵乘运算
    Matmul_out = Matmul_compute(W, Wz, Xin, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    Matmul_out = torch.tensor(Matmul_out)

    # # 5. 后面的FP模块的运算数据
    # # 一共3072*2=6144个FP16数据
    FP = Decode_txt_to_FP16(filename=data_dir+"saved_data/Fc2_FP.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    FPW = torch.tensor(FPW)
    FPB = torch.tensor(FPB)
    A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB), -128, 127)
    # print(A_out[0,0:100])

    """
    直接读取FPW和FPB的话,最终结果只有6个数出现偏差。而如果将它们
    先转为半精度浮点数,出现偏差的数据个数变成21184.
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
    FPB = zp2 + bias / Sx2 - 128                                                        

    # FPW = FPW.half().numpy()
    # FPB = FPB.half().numpy()
    FPW = FPW.numpy()
    FPB = FPB.numpy()
    FPW = torch.tensor(FPW)
    FPB = torch.tensor(FPB)
    A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB), -128, 127)
    # print(A_out[0,0:100])"""
    
    
    # 6. 读取真实的输出，看是否和A_out一致
    A_out_true_filename = f"{pth_prefix}fc2.act_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    A_out_true = A_out_true - 128
    # print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true).squeeze() 
    print("单层计算与真实值不相等的数据个数:", different_pos.shape[0])
    """
    实验结论: 基本上是一致的。有19390个数据不对。
    这是因为FPW和FPB先被转化为了FP16数据存在txt中, 从fp32到fp16产生了一定的精度损失。
    这是可以容忍的。如果像上面注释中直接读取FPW和FPB, 就会发现最终结果和test_fc1_out_simple()
    里面的结果一样, 只有2个数出现偏差。
    bigger和smaller都是0, 表示所有出现偏差的数据的数值范围都在1以内。
    """

    # 7. 存储矩阵乘运算结果
    # 这里我存两部分运算结果。
    # 第一部分是矩阵乘之后的结果, 第二部分是经过FP2之后的结果。
    # 前者是2048*48=98304行，512列。
    # 后者是2048*48=98304行，128列。
    # 值得说明的是我矩阵乘的结果还要重新算一遍，因为我得使用Layernorm+FP1的结果来进行矩阵乘，而不是从pth文件直接读
    # 的X_in，它们之间有+-1的误差。
    NUMBER_PER_LINE = args.array_size
    X_in = Decode_txt_to_int(filename=data_dir+"test_data/FP_quantizer1_out.txt", bit=8)
    X_in = X_in.reshape(args.vector_block, HEIGHT_BLOCK*args.array_size)
    Matmul_out = Matmul_compute(W, Wz, X_in, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    npy2hex2D(Matmul_out, data_dir+"test_data/Fc1_out.txt", 32, 0, NUMBER_PER_LINE, "w")

    FPW = FPW.numpy()
    FPB = FPB.numpy()

    A_out = np.clip(np.round(FPW * Matmul_out + FPB), -128, 127)      
    A_out = torch.tensor(A_out)
    different_pos = torch.nonzero(A_out!=A_out_true)
    print("累积计算与真实值不相等的数据个数: ", different_pos.shape[0])
    # print(A_out[0,0:100])   
    npy2hex2D(A_out, data_dir+"test_data/FP_quantizer2_out.txt", 8, 0, NUMBER_PER_LINE, "w") 
    """
    累积计算与真实值不相等的数据个数达到了115181个。
    bigger和smaller的结果分别为98, 34, 表示有数据的偏差在1以上了(最多有几个数偏差为3)。
    如果直接使用没有经过FP16保存的FPW和FPB, 则累积计算与真实值不相等的数据个数为111950, 也挺多。
    可见layernorm层后面的FP模块的输出偏差给后面的层带来了更多的偏差。而这一层使用fp16的FPW和FPB
    带来的误差有限。
    """
    return 

def test_fc2_out():
    # fc2的输出：int32
    # 经过第3个FP模块之后的输出：int 8
    # 我们比对FP之后的输出
    # 然后存储FC2的输出和FP之后的输出
    # 形状： [2048, 768]
    print("开始测试FC2层的输出。")
    HEIGHT_BLOCK = 48
    WIDTH_BLOCK = 12
    NUM_PER_LINE = args.array_size

    # 1. 读取txt文件，获得fc2的输入W 
    # 懒得把16进制数据一行行转为矩阵了，我就直接用W在存入Fc2_weight之前的数据
    weight_dequant_source_filename = f"{pth_prefix}fc2.weight_quantizer/dequant.pth"
    weight_dequant = torch.load(weight_dequant_source_filename, map_location=torch.device('cpu'))

    weight_scale_source_filename = f"{pth_prefix}fc2.weight_quantizer/scale.pth"
    weight_scale = torch.load(weight_scale_source_filename, map_location=torch.device('cpu'))

    weight_zp_source_filename = f"{pth_prefix}fc2.weight_quantizer/zeropoint.pth"
    weight_zp = torch.load(weight_zp_source_filename, map_location=torch.device('cpu'))

    bias_filename = f"{pth_prefix}fc2/bias.pth"
    bias = torch.load(bias_filename, map_location=torch.device('cpu')).detach()

    W = (weight_dequant.T / weight_scale + weight_zp)
    W = W.detach()
    W = torch.round(W).numpy()

    # 2. Wz, Xz
    # 存储格式为: 
    # Xz: 存了6行, 每行512个8bit有符号数
    # Wz: 存了2行，1024个8bit有符号数（只有前768个有效，后面的都是0）
    len_xz = args.ffn_size                                       # 这是Xz实际的长度。
    len_wz = args.hidden_size
    num_xz_per_line = args.weight_bram_width // args.x_bit
    num_wz_per_line = args.weight_bram_width // args.wz_bit
    len_xz_save = math.ceil(len_xz / num_xz_per_line) * num_xz_per_line   # 这是Xz在bram中存储的数据数，补了一些0
    len_wz_save = math.ceil(len_wz / num_wz_per_line) * num_wz_per_line
    Xz, Wz = Decode_txt_to_2int(
        filename=data_dir+"saved_data/Fc2_linear.txt",
        LEN_A = len_xz_save,
        LEN_B = len_wz_save,
        BIT_A = 8,
        BIT_B = 8,
        SIGN_A = True,
        SIGN_B = False
    )
    Xz = Xz[0 : len_xz]
    Wz = Wz[0 : len_wz]

    # 3. Fc1后面FP2输出的激活值
    # 我这里运算还是使用pth文件中的数据
    dequant_filename = f"{pth_prefix}fc2.act_quantizer/dequant.pth"
    dequant = torch.load(dequant_filename, map_location=torch.device('cpu')).detach()

    scale_filename = f"{pth_prefix}fc2.act_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zeropoint_filename = f"{pth_prefix}fc2.act_quantizer/zeropoint.pth"
    zeropoint = torch.load(zeropoint_filename, map_location=torch.device('cpu')).detach()

    Xin = (torch.clamp(torch.round(dequant / scale + zeropoint-128), -128, 127))
    Xin = Xin.detach().numpy()

    # 4. 进行矩阵乘运算
    Matmul_out = Matmul_compute(W, Wz, Xin, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    Matmul_out = torch.tensor(Matmul_out)

    # 5. 后面的FP模块的运算数据
    # 对于out_FP.txt: 一共是768*2个数据, 都是16bit。
    # 每行存了256个数据，所以共有6行。
    # 前面3行是W，后面3行是B.
    # 还有支路的数据: skip, Layernorm_in
    FP = Decode_txt_to_FP16(filename=data_dir+"saved_data/out_FP.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    FPW = torch.tensor(FPW)
    FPB = torch.tensor(FPB)

    s_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    s = torch.load(s_filename, map_location=torch.device('cpu')).item()
    s_next_block_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    s_next_block = torch.load(s_next_block_filename, map_location=torch.device('cpu')).item()
    skip = s / s_next_block

    LN_Xin = Decode_txt_to_int(filename=data_dir+'saved_data/Layernorm_Xin.txt', bit=8)
    LN_Xin = LN_Xin.reshape(args.vector_block, WIDTH_BLOCK*args.array_size)

    A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB + LN_Xin * skip), -128, 127)

    A_out_true_filename = f"{pth_next_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    # print(A_out[0,0:100])
    # print(A_out_true[0,0,0:100])
    diffrent_pos = torch.nonzero(A_out != A_out_true)
    print("单层计算与真实值不相等的数据个数为: ", diffrent_pos.shape[0])
    """
    实验结论: A_out和A_out_true基本上是对的。没问题。差了1832个数据。
    """
    # 6. 创建累积计算输出。这个输出实际上是下一个layer的attention模块的量化后输入
    # 这里使用的数据一定要是FC1后面的那个FP模块给出的数据.
    # 存两部分结果。
    # 第一部分是矩阵乘之后的结果，为int32，一共12*2048=24567行，512列。
    # 第二部分是加上残差连接然后FP之后的结果，为int8，一共12*2048=24567行，128列。
    X_in = Decode_txt_to_int(filename=data_dir+"test_data/FP_quantizer2_out.txt", bit=8, sign=True)
    X_in = X_in.reshape(args.vector_block, HEIGHT_BLOCK*args.array_size)

    Matmul_out = Matmul_compute(W, Wz, X_in, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    npy2hex2D(Matmul_out, data_dir+"test_data/Fc2_out.txt", 32, 0, NUM_PER_LINE, "w")

    FPW = FPW.numpy().astype(np.float16)
    FPB = FPB.numpy().astype(np.float16)
    skip = np.array(skip, dtype=np.float16)
    A_out = np.clip(np.round(FPW * Matmul_out + FPB + LN_Xin * skip), -128, 127)
    diffrent_pos = torch.nonzero(torch.tensor(A_out) != A_out_true)
    print("累积计算与真实值不相等的数据个数为: ", diffrent_pos.shape[0])
    """
    实验结论: 差了245715个数据。
    """
    # np.save('FPW.npy', FPW)
    # np.save('FPB.npy', FPB)
    # np.save('Matmul_out.npy', Matmul_out)
    # np.save('LN_Xin.npy', LN_Xin)
    # np.save('skip.npy', skip)
    # np.save('A_out.npy', A_out)
    # print(torch.tensor(A_out[0,0:100]))
    npy2hex2D(A_out, data_dir+"test_data/FP_quantizer3_out.txt", 8, 0, NUM_PER_LINE, "w")
    return 

def trans_txt_to_mem(source, dest):
    with open(source, "r") as txt:
        with open(dest, "w") as mem:
            address = 0
            for line in txt:
                mem.write(f"@{address:08X} {line}")
                address += 1

def llm_test_fc_data():
    # 1. 存Layernorm之后的输出和第一次FP之后的输出
    # test_layernorm_out()

    # 2. 存FC1之后的输出和第二次FP之后的输出
    test_fc1_out()

    # 3. 存FC2之后的输出和第三次FP之后的输出
    test_fc2_out()
    
    return 

if __name__ == '__main__':
    llm_test_fc_data()