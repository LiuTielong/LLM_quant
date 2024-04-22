"""
做三件事：
1. 将layer0的Attention block的一些中间层状态存下来, 存为txt文件。
2. 验证create_attn_data.py里面生成的mem文件是否存储正确。
3. 用python模拟跑一遍Attention block的数据流。
还需要加的东西: 现在的KV cache都是空的, 未来加上KV cache之后再做验证应该难度系数更高。
"""
import torch
import numpy as np
import math
import struct
from llm_create_fc_data import (
    npy2hex,
    npy2hex2D,
    npy2hex_fp16,
    data2hex
)
from llm_test_fc_data import (
    Decode_txt_to_int, 
    trans_txt_to_mem,
    I_SQRT,
    Matmul_compute,
    Decode_txt_to_2int,
    Decode_txt_to_FP16,
    de_reorder_fp_data
    )
from llm_create_attn_data import(
    HEIGHT_BLOCK,
    WIDTH_BLOCK,
)
import pdb

from llm_args import args
data_dir = args.data_dir_attn
pth_prefix = args.pth_prefix

def fixed(x, total_bit, frac_bit):
    x_max = 2 ** (total_bit - 1) - 1
    x_min = -2 ** (total_bit - 1)
    x_lshift = np.clip(np.round(x * 2**frac_bit), x_min, x_max)
    return  x_lshift / 2 **frac_bit

def test_layernorm_out():
    # layernorm的输出: int 32
    # 经过FP之后, 为int 8.
    # 我要测试的就是经过FP模块之后的结果
    # 形状: [2048, 768]
    print("开始测试layernorm的输出结果。")
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

    # 3. 最后的浮点数乘法
    FP = Decode_txt_to_FP16(data_dir +'saved_data/Layernorm_out_fp.txt')
    FPW, FPB = de_reorder_fp_data(FP)
    y = np.clip(np.round(FPW * x_fn + FPB), -128, 127)       

    """ 
    # 备注: 将FPW和FPB存为fp16后再取出来计算, 产生的代价太大了。
    # 不相等的数据个数从21增加到了24116. 
    ln_wd2m_source_filename = f"{pth_prefix}self_attn_layer_norm/w_div_2M.pth"
    ln_wd2m = torch.load(ln_wd2m_source_filename, map_location='cpu').detach()

    ln_bias_filename = f"{pth_prefix}self_attn_layer_norm/bias.pth"
    ln_bias = torch.load(ln_bias_filename, map_location=torch.device('cpu')).detach()   

    scale_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/scale.pth"
    scale = torch.load(scale_filename, map_location=torch.device('cpu')).detach()

    zero_point_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/zeropoint.pth"
    zero_point = torch.load(zero_point_filename, map_location=torch.device('cpu')).detach()

    ln_wd2m_fusion = ln_wd2m / scale
    ln_bias_fusion = ln_bias / scale + zero_point
    ln_bias_fusion -= args.x_shift
    y = np.clip(np.round(ln_wd2m_fusion * x_fn + ln_bias_fusion), -128, 127)  """

    # print(y[0,0:100])

    # 4. 接下来读取真实的输出结果
    QKV_in_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/x_int_add_zp_clamp.pth"
    QKV_in = torch.load(QKV_in_filename, map_location='cpu').detach().squeeze() - 128
    QKV_in = torch.round(QKV_in)
    # print(QKV_in[0,0:100])

    # 5. 比较不同
    different_pos = torch.nonzero(torch.tensor(y) != QKV_in)
    print("不相等的数据个数为:", different_pos.shape[0])
    """
    实验结论: 有24116多个数据对不上。据我观察, 这些数据的偏差都是在+-1.
    主要的误差来源于对FPW和FPB的存储后再读!
    """

    # 6. 存储运算结果. 这里我存layernorm的输出以及第一个FP的输出
    # layernorm的输出：32bit。每行存64个数，共计2048*12=24576行
    npy2hex2D(x_fn, data_dir+"test_data/Layernorm_Xout.txt", 32, 0, NUM_PER_LINE, "w")
    # 第一个FP的输出: 8bit, 每行存64个数，共计2048*12=24576行
    npy2hex2D(y, data_dir+"test_data/Layernorm_out_quantizer_out.txt", 8, 0, NUM_PER_LINE, "w")
    return 

def test_qkv_out(name="q_proj"):
    # 测试QKV三个矩阵分别的输出是否正确。
    # 重要问题: 从pth文件读取的W实际上还是按照width * height的方式存储的, 所以我需要先将它
    # 转置一下。
    NUMBER_PER_LINE = args.array_size
    print(f"开始测试{name}层的输出。")
    
    # 1. 读取W
    W_filename = f"{pth_prefix}self_attn.{name}.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location=torch.device('cpu')).detach()
    W = torch.round(W).numpy().T

    # 2. Wz, Xz
    len_xz = args.hidden_size                                               # 这是Xz实际的长度。
    len_wz = args.hidden_size
    num_xz_per_line = args.weight_bram_width // args.x_bit
    num_wz_per_line = args.weight_bram_width // args.wz_bit
    len_xz_save = math.ceil(len_xz / num_xz_per_line) * num_xz_per_line     # 这是Xz在bram中存储的数据数，补了一些0
    len_wz_save = math.ceil(len_wz / num_wz_per_line) * num_wz_per_line
    Xz, Wz = Decode_txt_to_2int(
        filename=data_dir+"saved_data/"+name+"_linear.txt",
        LEN_A = len_xz_save,
        LEN_B = len_wz_save,
        BIT_A = 8,
        BIT_B = 8,
        SIGN_A = True,
        SIGN_B = False
    )
    Xz = Xz[0 : len_xz]
    Wz = Wz[0 : len_wz]

    # 3. layernorm后面的FP的输出值
    # 先确保这一层的运算正确，所以就加载pth文件中的数据。
    # 等这一层运算正确之后再换test_layernorm_out()的输出结果。
    Xin_filename = f"{pth_prefix}self_attn_layer_norm.out_quantizer/x_int_add_zp_clamp.pth"
    Xin = torch.load(Xin_filename, map_location=torch.device('cpu')).detach()
    Xin = torch.round(Xin[0]).numpy() - args.x_shift
    
    # 4. 进行矩阵乘运算
    Matmul_out = Matmul_compute(W, Wz, Xin, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    Matmul_out = torch.tensor(np.round(Matmul_out))

    # 5. 后面的FP模块的运算数据
    # 各有768个数据
    FP = Decode_txt_to_FP16(filename=data_dir+"saved_data/"+name+"_out_fp.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    FPW = torch.tensor(FPW)
    FPB = torch.tensor(FPB)

    """ 如果直接计算FPW和FPB, 而不是先存储为FP16再读取的话, 最终结果只会有
    几个数和真实值不同。
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
        FPW = Sx1 * Sw1 / Sx2 / 8
        FPB = torch.round(Zp2 - args.x_shift) + bias / Sx2 / 8
    else :
        FPW = Sx1 * Sw1/ Sx2
        FPB = torch.round(Zp2) + bias / Sx2
    """

    if name == "q_proj":
        A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB), -128, 127)
    else :
        A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB), 0, 255)

    # 6. 读取真实的输出, 看是否和A_out一致
    if name == "q_proj":
        A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    elif name == "k_proj":
        A_out_true_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    elif name == "v_proj":
        A_out_true_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    if name == "q_proj":
        A_out_true = A_out_true.squeeze() - 128
    
    # print(A_out[0,0:100])
    # print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true).squeeze() 
    print("单层计算导致不相等的数据个数为: ", different_pos.shape)
    """
    实验结论: 基本上能对得上。但由于浮点数精度问题，有些位置上会有误差。
    具体来说是先将FPW和FPB存为16bit再读出来造成了较大的损失。
    对于q_prj: 有5803个数出现偏差, 基本上偏差都在+-1。
    对于k_proj: 有26101个数出现偏差。
    对于v_proj: 有37225个数出现偏差。
    """

    # 7. 使用self_attn_layernorm后面的FP的输出作为输入, 获取矩阵乘结果存入txt文件。
    X_in = Decode_txt_to_int(filename=data_dir + "test_data/Layernorm_out_quantizer_out.txt", bit=8, sign=True)
    X_in = X_in.reshape(args.vector_block, HEIGHT_BLOCK*args.array_size)
    Matmul_out = Matmul_compute(W, Wz, X_in, Xz, 
                                    VECTOR_BLOCK=args.vector_block, 
                                    HEIGHT_BLOCK=HEIGHT_BLOCK,
                                    WIDTH_BLOCK=WIDTH_BLOCK,
                                    BIT_BLOCK=args.weight_bit_block)
    npy2hex2D(Matmul_out, data_dir+f"test_data/{name}_out.txt", 32, 0, NUMBER_PER_LINE, "w")

    FPW = FPW.numpy()
    FPB = FPB.numpy()
    if name == "q_proj":
        A_out = np.clip(np.round(Matmul_out * FPW + FPB), -128, 127)
    else :
        A_out = np.clip(np.round(Matmul_out * FPW + FPB), 0, 255)
    A_out = torch.tensor(A_out)
    different_pos = torch.nonzero(A_out!=A_out_true)
    # largger = torch.sum(A_out > A_out_true+1)
    print("累积计算导致不相等的数据个数为: ", different_pos.shape[0])   
    npy2hex2D(A_out.numpy(), data_dir+f"test_data/{name}_fp_out.txt", 8, 0, NUMBER_PER_LINE, "w")
    """
    备注: 对于q_proj, 产生偏差的数据达到176789.     
    对于k_proj, 产生偏差的数据达到214960个。         
    对于v_proj, 产生偏差的数据达到310954个。        
    说实话这个数据量已经是不能忍了。             
    不过我又发现这些出现偏差的数据里面, 比真实值大1和小1的数据个数差不多。
    而且没有出现和真实值差距在1以上的数据。   
    """
    return


def test_qkt_matmul_out():
    # 先进行单层测试
    # 1. 直接读取Q, K, Q_zp, K_zp
    print("开始测试QKT_matmul的输出。")
    Q_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    Q = torch.load(Q_filename, map_location=torch.device('cpu')).detach().squeeze()

    K_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    K = torch.load(K_filename, map_location=torch.device('cpu')).detach().squeeze()

    Q_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/zeropoint.pth"
    Q_zp = torch.load(Q_zp_filename, map_location=torch.device('cpu')).detach() - args.x_shift

    K_zp_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/zeropoint.pth"
    K_zp = torch.load(K_zp_filename, map_location=torch.device('cpu')).detach()  # 1个数

    Q = torch.round(Q) - args.x_shift
    K = torch.round(K)
    Q = Q.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)
    K = K.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)
    KT = K.transpose(1, 2)
    Q = Q.numpy()
    KT = KT.numpy()
    Q_zp = torch.round(Q_zp).numpy()            # 这两个zeropoint不是从txt文件中读的，方便。
    K_zp = torch.round(K_zp).numpy()            # 反正它们都是一个数

    Matmul_out = np.zeros((args.num_head, args.vector_block, args.vector_block))
    for head_index in range(args.num_head):
        Matmul_out[head_index] = Matmul_compute(
            KT[head_index], K_zp, Q[head_index], Q_zp,
            VECTOR_BLOCK=args.vector_block, 
            HEIGHT_BLOCK = args.head_size // args.array_size,
            WIDTH_BLOCK=args.vector_block // args.array_size,
            BIT_BLOCK=args.x_bit
        )
    Matmul_out = torch.tensor(np.round(Matmul_out))
    # Matmul_out = torch.tensor(np.matmul(Q - Q_zp, KT - K_zp)) 

    # 2. 读取FP单元的参数
    # (1) 直接读取pth文件中的参数，只有2个数据对不上。
    # Sq_filename = f"{pth_prefix}self_attn.qkt_matmul.x1_quantizer/scale.pth"
    # Sq = torch.load(Sq_filename, map_location=torch.device('cpu')).detach()

    # Sk_filename = f"{pth_prefix}self_attn.qkt_matmul.x2_quantizer/scale.pth"
    # Sk = torch.load(Sk_filename, map_location=torch.device('cpu')).detach()

    # Sout_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/scale.pth"
    # Sout = torch.load(Sout_filename, map_location=torch.device('cpu')).detach()

    # Zpout_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/zeropoint.pth"
    # Zpout = torch.load(Zpout_filename, map_location=torch.device('cpu')).detach()

    # A_out = torch.clamp(torch.round(Matmul_out * Sq * Sk / Sout + Zpout), -128, 127)

    # (2) 从我存储的txt文件再转换出fpw和fpb，有31383个数据对不上。
    FP = Decode_txt_to_FP16(filename=data_dir+"saved_data/qkt_out_fp.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    A_out = torch.clamp(torch.round(Matmul_out * FPW[0] + FPB[0]), -128, 127)

    # 3. 读取真实值
    A_out_true_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach()
    diffrent_pos = torch.nonzero(A_out != A_out_true)
    print("单层计算导致的不正确的数据个数为:", diffrent_pos.shape)

    # 4.计算多层累积结果及累积误差
    Q = Decode_txt_to_int(filename=data_dir + "test_data/q_proj_fp_out.txt", bit=8, sign=True)
    K = Decode_txt_to_int(filename=data_dir + "test_data/k_proj_fp_out.txt", bit=8, sign=False)
    Q = torch.tensor(np.round(Q))
    K = torch.tensor(np.round(K))
    Q = Q.reshape(args.vector_block, args.hidden_size)
    K = K.reshape(args.vector_block, args.hidden_size)
    Q = Q.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)
    K = K.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)
    KT = K.transpose(1, 2)
    Q = Q.numpy()
    KT = KT.numpy()

    Matmul_out = np.zeros((args.num_head, args.vector_block, args.vector_block))
    for head_index in range(args.num_head):
        Matmul_out[head_index] = Matmul_compute(
            KT[head_index], K_zp, Q[head_index], Q_zp,
            VECTOR_BLOCK=args.vector_block, 
            HEIGHT_BLOCK = args.head_size // args.array_size,
            WIDTH_BLOCK=args.vector_block // args.array_size,
            BIT_BLOCK=args.x_bit
        )
    Matmul_out = torch.tensor(np.round(Matmul_out))
    A_out = torch.clamp(torch.round(Matmul_out * FPW[0] + FPB[0]), -128, 127)
    diffrent_pos = torch.nonzero(A_out != A_out_true)
    print("累积计算导致的不正确的数据个数为:", diffrent_pos.shape)  
    """
    实验结果: 出现5790849个数据出现问题, 太不能忍了！！！
    数据总数是50331648, 大约是12%.
    """
    # 5. 分head进行存储
    Matmul_out = Matmul_out.numpy()
    A_out = A_out.numpy()
    for head_index in range(args.num_head):
        if head_index == 0:
            npy2hex2D(Matmul_out[head_index], data_dir+f"test_data/qkt_matmul_out.txt", 32, 0, args.array_size, "w")
            npy2hex2D(A_out[head_index], data_dir+f"test_data/qkt_matmul_fp_out.txt", 8, 0, args.array_size, "w")
        else:
            npy2hex2D(Matmul_out[head_index], data_dir+f"test_data/qkt_matmul_out.txt", 32, 0, args.array_size, "a")
            npy2hex2D(A_out[head_index], data_dir+f"test_data/qkt_matmul_fp_out.txt", 8, 0, args.array_size, "a")
    return


def test_softmax_out():
    print("开始测试Softmax的输出。")
    # 先进行单层测试
    # 1. 直接读取softmax的输入: QKT matmul的结果
    # t1_source_filename = f"{pth_prefix}self_attn.softmax/t1.pth"
    # t1 = torch.load(t1_source_filename, map_location=torch.device('cpu')).item()
    # t1_hex = data2hex(t1, 16, 16)   

    # t2_source_filename = f"{pth_prefix}self_attn.softmax/t2.pth" 
    # t2 = torch.load(t2_source_filename, map_location=torch.device('cpu')).item()   
    # t2_hex = data2hex(t2, 16, 8)

    # t3_source_filename = f"{pth_prefix}self_attn.softmax/t3.pth" 
    # t3 = torch.load(t3_source_filename, map_location=torch.device('cpu')).item()
    # t3_hex = data2hex(t3, 32, 8)   
    # print(t1)
    # print(t2)
    # print(t3)
    # print(t1_hex)
    # print(t2_hex)
    # print(t3_hex)

    t1 = 4 / 16 + 13 / 16 **2 + 3 / 16 ** 3 + 4 / 16 ** 4
    t2 = 3  + 5 / 16 + 1 / 16**2 
    t3 = 9 + 9 / 16 + 1 / 16**2 
    # 1. 读取txt文件，获得LN的输入
    # Softmax_Xin_filename = f"{pth_prefix}self_attn.softmax.in_quantizer/x_int_add_zp_clamp.pth"
    # Softmax_Xin = torch.load(Softmax_Xin_filename, map_location="cpu").detach()
    # Softmax_Xin = torch.round(Softmax_Xin).numpy()
    Softmax_Xin = Decode_txt_to_int(data_dir+f"test_data/qkt_matmul_fp_out.txt", 8, True)
    Softmax_Xin = torch.tensor(np.round(Softmax_Xin))
    Softmax_Xin = Softmax_Xin.reshape(args.num_head, args.vector_block, args.vector_block).numpy()

    # 1.5 首先要进行mask操作
    # 先加上mask，然后不能比-128小，把这个作为softmax真实的输入
    Mask = np.triu(-255 * np.ones((2048, 2048)), k=1)
    Softmax_Xin = np.round(Softmax_Xin + Mask)
    Softmax_Xin = np.clip(Softmax_Xin, -128, 127)
    # 这个mask操作我挪到v后面的话，就产生很大的误差！

    # 2. 模拟softmax运算
    # 2.1 
    xm = np.max(Softmax_Xin, axis=-1, keepdims=True)
    # xm = torch.tensor(np.round(xm))
    # xm_true_filename = f"{pth_prefix}self_attn.softmax/x_int_max.pth"
    # xm_true = torch.load(xm_true_filename, map_location="cpu").detach()
    # xm_true = torch.round(xm_true)
    # different_pos = torch.nonzero(xm != xm_true)
    # print("不同的个数:", different_pos.shape)
    # print(xm[0,35,0], xm_true[0,35,0])
    # 没问题

    # 2.2
    x_fix = np.round(Softmax_Xin - xm)
    # x_fix = torch.tensor(x_fix)
    # x_fix_true_filename = f"{pth_prefix}self_attn.softmax/x_int_sub_max.pth"
    # x_fix_true = torch.load(x_fix_true_filename, map_location="cpu").detach()
    # x_fix_true = torch.round(x_fix_true)
    # different_pos = torch.nonzero(x_fix != x_fix_true)
    # print(x_fix[0,0,1], x_fix_true[0,0,1])
    # print("不同的个数:", different_pos.shape)

    z = np.floor(- x_fix * t1)
    p = x_fix + t2 * z + t3
    p = fixed(p, 32, 24)
    v = p / 2 ** z
    v = fixed(v, 32, 16)
    Mask = np.tril(1 * np.ones((2048, 2048)), k=0)
    v = v * Mask

    # 2.3
    v = torch.tensor(v)
    # v_true_filename = f"{pth_prefix}self_attn.softmax/v.pth"
    # v_true = torch.load(v_true_filename, map_location="cpu").detach()
    # different_pos = torch.nonzero(v!= v_true)
    # print(different_pos.shape)
    # v = v.to(torch.float32)
    # different_pos = torch.nonzero(v!= v_true)
    # print(different_pos.shape)
    # 完全相等

    # 2.4
    v_sum = torch.sum(v.to(torch.float32), dim=-1, keepdims=True)
    v_sum = v_sum.numpy()
    v_sum = fixed(v_sum, 32, 16)
    # v_sum_true_filename = f"{pth_prefix}self_attn.softmax/v_sum.pth"
    # v_sum_true = torch.load(v_sum_true_filename, map_location="cpu").detach()
    # different_pos = torch.nonzero(abs(v_sum- v_sum_true)>0.0001)
    # print(different_pos.shape)
    # 阈值设为0.001的话就没有问题，如果设置成0.0001的话就有909个数对不上

    # 2.5
    v = v.numpy()
    # v_shift = np.round(2 ** 16 * v)
    v_shift = v
    # v_shift = v_shift - 2**16 * np.floor(v_shift/ 2**16)    # 模拟了硬件中向左移位过程中舍掉的高位
    y = v_shift * 2 **(args.softmax_m)/ v_sum
    y = fixed(y, 32, 16)
    # y = torch.tensor(y)
    # y_true_filename = f"{pth_prefix}self_attn.softmax/result.pth"
    # y_true = torch.load(y_true_filename, map_location="cpu").detach()
    # different_pos = torch.nonzero(abs(y- y_true)>0.0001)
    # print(different_pos.shape)
    # 完全相等

    # 3. 进行quantize
    # 读取我保存的txt文件中的FPW和FPB
    FP = Decode_txt_to_FP16(data_dir+"saved_data/Softmax_out_fp.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    # 直接读取FPW和FPB的结果: 完全对得上
    # scale_out_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/scale.pth"
    # scale_out = torch.load(scale_out_filename, map_location="cpu").detach()
    # FPW = torch.tensor(1 / scale_out)
    # FPB = - 128

    Softmax_Xout = np.clip(np.round(FPW * y + FPB), -128, 127)

    # 3. 对比softmax的真实输出
    Softmax_out_true_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    Softmax_out_true = torch.load(Softmax_out_true_filename, map_location=torch.device('cpu')).detach() - 128
    Softmax_out_true = torch.round(Softmax_out_true).numpy()
    # print(Softmax_Xout[0, 20, 0:20])
    # print(Softmax_out_true[0, 20, 0:20])

    Softmax_out_true = torch.tensor(Softmax_out_true)
    Softmax_Xout = torch.tensor(Softmax_Xout).round()
    different_pos = torch.nonzero(Softmax_Xout != Softmax_out_true)
    print("算出来的softmax有偏差的数据个数为:", different_pos.shape[0])
    """
    实验结论: 
    单层计算: 如果直接从pth文件读取FPW和FPB, 完全没有误差。
    如果从我保存的txt文件读取FPW和FPB, 有352个数有误差, 还能接受。
    累积计算: 出现误差的数据达到141827个。还是有些误差的。
    """

    # 4. 累积计算输出
    Softmax_Xout = Softmax_Xout.numpy()
    for head_index in range(args.num_head):
        if head_index == 0:
            # npy2hex2D(xm[head_index], data_dir+'softmax_stage1_out.txt', 16, 0, 1, "w")
            # npy2hex2D(v[head_index], data_dir+'softmax_stage2_out_v.txt', 32, 16, 64, "w")
            # npy2hex2D(y[head_index], data_dir+'softmax_stage3_out.txt', 32, 16, 64, "w")
            npy2hex2D(Softmax_Xout[head_index], data_dir+"test_data/softmax_fp_out.txt", 8, 0, 64, "w")
        else:
            # npy2hex2D(xm[head_index], data_dir+'softmax_stage1_out.txt', 16, 0, 1, "a")
            # npy2hex2D(v[head_index], data_dir+'softmax_stage2_out_v.txt', 32, 16, 64, "a")
            # npy2hex2D(y[head_index], data_dir+'softmax_stage3_out.txt', 32, 16, 64, "a")
            npy2hex2D(Softmax_Xout[head_index], data_dir+"test_data/softmax_fp_out.txt", 8, 0, 64, "a")
    return 


def test_pv_matmul_out():
    # 1. 单层测试, 先读取P和V
    print("开始测试pv_matmul的输出。")
    # P_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/x_int_add_zp_clamp.pth"
    # P = torch.load(P_filename, map_location=torch.device('cpu')).detach()
    # V_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/x_int_add_zp_clamp.pth"
    # V = torch.load(V_filename, map_location=torch.device('cpu')).detach()
    # P = torch.round(P) - args.x_shift
    # V = torch.round(V)
    # V = V.view(args.vector_block, args.num_head, args.head_size).transpose(0, 1)
    # P = P.numpy()
    # V = V.numpy()
    P = Decode_txt_to_int(data_dir+f"test_data/softmax_fp_out.txt", bit=8, sign=True)
    V = Decode_txt_to_int(data_dir+f"test_data/v_proj_fp_out.txt", bit=8, sign=False)
    P = P.reshape(args.num_head, args.vector_block, args.vector_block)
    V = V.reshape(args.vector_block, args.num_head, args.head_size).swapaxes(0, 1)


    # P_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/zeropoint.pth"
    # P_zp = torch.load(P_zp_filename, map_location=torch.device('cpu')).detach() - args.x_shift
    # V_zp_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/zeropoint.pth"
    # V_zp = torch.load(V_zp_filename, map_location=torch.device('cpu')).detach()
    # P_zp = P_zp.numpy()
    # V_zp = V_zp.numpy()
    P_zp, V_zp = Decode_txt_to_2int(
        data_dir+"saved_data/pv_linear.txt",
        LEN_A = args.vector_block,
        LEN_B = args.hidden_size,
        BIT_A = 8,
        BIT_B = 8,
        SIGN_A = True,
        SIGN_B = False
    )
    P_zp = P_zp[0]

    # 2. 矩阵乘运算，没问题
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

    # 3. 读取FP单元的参数，进行量化操作
    # Sp_filename = f"{pth_prefix}self_attn.pv_matmul.x1_quantizer/scale.pth"
    # Sp = torch.load(Sp_filename, map_location=torch.device('cpu')).detach()
    # Sv_filename = f"{pth_prefix}self_attn.pv_matmul.x2_quantizer/scale.pth"
    # Sv = torch.load(Sv_filename, map_location=torch.device('cpu')).detach()
    # Sout_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    # Sout = torch.load(Sout_filename, map_location=torch.device('cpu')).detach()
    # Zpout_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    # Zpout = torch.load(Zpout_filename, map_location=torch.device('cpu')).detach() - args.x_shift
    # A_out = torch.clamp(torch.round(Matmul_out * Sp * Sv / Sout + Zpout), -128, 127)

    FP = Decode_txt_to_FP16(filename=data_dir+"saved_data/pv_matmul_out_fp.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    FPW = torch.tensor(FPW)
    FPB = torch.tensor(np.round(FPB))
    A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB), -128, 127)

    # 4. 加载真实值进行比较
    A_out_true_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location=torch.device('cpu')).detach() - args.x_shift
    A_out_true = torch.round(A_out_true[0])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("不相等的数据个数为:", different_pos.shape[0])
    """
    实验结论: 
    1. 单层计算: 读取pth文件的S和Zp, 误差数为1.
    2. 单层计算: 读取txt文件的FPW和FPB以及Zp, 误差数为1385, 也不大。
    3. 多层累积计算: 误差数为255000个, 肉眼看差距不大。
    """

    # 5. 输出并保存结果
    # print(A_out[0,0:100])
    # print(A_out_true[0,0:100])
    npy2hex2D(A_out, data_dir+"test_data/pv_matmul_fp_out.txt", 8, 0, num_per_line=args.array_size, type="w")
    return


def test_out_proj_out():
    print("开始测试out_proj的输出。")
    # 1. 单层测试, 先读取W, W_zp, Xin, Xin_zp
    W_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/x_int_add_zp_clamp.pth"
    W = torch.load(W_filename, map_location="cpu").detach()
    W = torch.round(W.T)
    W = W.numpy()
    # Xin_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/x_int_add_zp_clamp.pth"
    # Xin = torch.load(Xin_filename, map_location="cpu").detach()
    # X_zp_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/zeropoint.pth"
    # X_zp = torch.load(X_zp_filename, map_location="cpu").detach()
    # W_zp_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/zeropoint.pth"
    # W_zp = torch.load(W_zp_filename, map_location="cpu").detach()
    # W_zp = W_zp.reshape(1, -1)
    # W_zp = torch.round(W_zp).numpy()
    # Xin = torch.round(Xin[0]).numpy()
    # X_zp = torch.round(X_zp).numpy()
    len_xz = args.hidden_size                                               # 这是Xz实际的长度。
    len_wz = args.hidden_size
    num_xz_per_line = args.weight_bram_width // args.x_bit
    num_wz_per_line = args.weight_bram_width // args.wz_bit
    len_xz_save = math.ceil(len_xz / num_xz_per_line) * num_xz_per_line     # 这是Xz在bram中存储的数据数，补了一些0
    len_wz_save = math.ceil(len_wz / num_wz_per_line) * num_wz_per_line
    X_zp, W_zp = Decode_txt_to_2int(
        filename=data_dir+"saved_data/out_proj_linear.txt",
        LEN_A = len_xz_save,
        LEN_B = len_wz_save,
        BIT_A = 8,
        BIT_B = 8,
        SIGN_A = True,
        SIGN_B = False
    )
    X_zp = X_zp[0:args.hidden_size]
    W_zp = W_zp[0:args.hidden_size]
    Xin = Decode_txt_to_int(filename=data_dir+"test_data/pv_matmul_fp_out.txt", bit=8, sign=True)
    Xin = Xin.reshape(args.vector_block, args.hidden_size)

    # 2. 矩阵乘法
    Matmul_out = Matmul_compute(
        W, W_zp, Xin, X_zp, 
        VECTOR_BLOCK =  args.vector_block,
        HEIGHT_BLOCK = args.hidden_size // args.array_size, 
        WIDTH_BLOCK = args.hidden_size // args.array_size, 
        BIT_BLOCK = args.weight_bit_block
    )

    # 3. 加载FPW，FPB，skip，LN_Xin
    LN_Xin_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    LN_Xin = torch.load(LN_Xin_filename, map_location="cpu").detach()
    LN_Xin = torch.round(LN_Xin)
    Sx_ln_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/scale.pth"
    Sx_ln = torch.load(Sx_ln_filename, map_location=torch.device('cpu')).detach()
    Sx_next_filename = f"{pth_prefix}final_layer_norm.in_quantizer/scale.pth"
    Sx_next = torch.load(Sx_next_filename, map_location=torch.device('cpu')).detach()
    skip = Sx_ln / Sx_next

    # W_scale_filename = f"{pth_prefix}self_attn.out_proj.weight_quantizer/scale.pth"
    # W_scale = torch.load(W_scale_filename, map_location="cpu").detach()
    # Bias_filename = f"{pth_prefix}self_attn.out_proj/bias.pth"
    # Bias = torch.load(Bias_filename, map_location="cpu").detach()
    # Zp_ln_filename = f"{pth_prefix}self_attn_layer_norm.in_quantizer/zeropoint.pth"
    # Zp_ln = torch.load(Zp_ln_filename, map_location=torch.device('cpu')).detach()
    # In_scale_filename = f"{pth_prefix}self_attn.out_proj.act_quantizer/scale.pth"
    # In_scale = torch.load(In_scale_filename, map_location="cpu").detach()
    # FPW = In_scale * W_scale / Sx_next
    # FPB = Bias / Sx_next + Sx_ln * Zp_ln / Sx_next
    # Matmul_out = torch.tensor(np.round(Matmul_out))
    # A_out = torch.clamp(torch.round(Matmul_out * FPW + FPB + LN_Xin * skip), -128, 127)

    skip = np.array(skip, dtype=np.float16)
    LN_Xin = LN_Xin[0].numpy()
    FP = Decode_txt_to_FP16(data_dir+"saved_data/out_proj_out_FP.txt")
    FPW, FPB = de_reorder_fp_data(FP)
    A_out = np.clip(np.round(Matmul_out * FPW + FPB + LN_Xin * skip), -128, 127)
    A_out = torch.tensor(A_out)

    # 4. 读取真实输出进行比较
    A_out_true_filename = f"{pth_prefix}final_layer_norm.in_quantizer/x_int_add_zp_clamp.pth"
    A_out_true = torch.load(A_out_true_filename, map_location="cpu").detach()
    # print(A_out[0,0:100])
    # print(A_out_true[0,0:100])
    different_pos = torch.nonzero(A_out != A_out_true)
    print("不相等的数据个数为:", different_pos.shape)

    npy2hex2D(A_out.numpy(), data_dir+"test_data/out_proj_fp_out.txt", 8, 0, args.array_size, "w")
    """
    实验结论: 
    1. 单层测试: 从pth文件生成FPW和FPB, 出现错误的数据个数为: 6。微乎其微。
    2. 单层测试: 从txt文件生成FPW和FPB,以及把skip置为fp16, 出现错误的数据个数为: 2713。
    3. 累积测试: 将矩阵乘的输入换成上一层的输出。Zeropoint要从txt文件读取。测试结果有477163个数出现偏差, 接近1/3了。
    肉眼看上去偏差范围都在+-1. 接下来就是考虑如何优化了。
    """
    return

def llm_test_attn_data():
    # 测试顺序为: 
    # 1. attn_layernorm的输出
    test_layernorm_out()

    # 2. q_proj, k_proj, v_proj的输出
    test_qkv_out(name="q_proj")
    test_qkv_out(name="k_proj")
    test_qkv_out(name="v_proj")

    # 3. matmul(q, k.T)的输出
    test_qkt_matmul_out()

    # 4. softmax的输出
    test_softmax_out()

    # 5. matmul(p,v)的输出
    test_pv_matmul_out()

    # 6. out_proj的输出
    test_out_proj_out()

    return 

if __name__ == "__main__":
    llm_test_attn_data()