import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Layer args')
    parser.add_argument("--data_dir", type=str, 
                        default="../llm_all_data/", 
                        # default="F:/created_data_4_16_fp16/",
                        help="where to save created data")
    parser.add_argument("--pth_pre", type=str, 
                        # default="../tensors/model.decoder.layers."
                        default="F:/layer0_4_16/model.decoder.layers."
                        # default="F:/layer0_4_16_fp16/model.decoder.layers."
                        )
    parser.add_argument('--layer_num', type=int, default=0, help='layer number')
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size")
    parser.add_argument("--ffn_size", type=int, default=3072, help="ffn size")
    parser.add_argument("--head_size", type=int, default=64)
    parser.add_argument("--num_head", type=int, default=12)
    parser.add_argument("--vector_block", type=int, default=2048)
    parser.add_argument("--weight_bit_block", type=int, default=2, help="weight bit block")
    parser.add_argument("--wz_bit", type=int, default=8, help="wz bit")
    parser.add_argument("--x_bit", type=int, default=8, help="x bit = xz bit")
    parser.add_argument("--x_shift", type=int, default=128, help="trans uint8 to int8")
    parser.add_argument("--array_size", type=int, default=64)
    parser.add_argument("--weight_bram_width", type=int, default=4096)
    parser.add_argument("--num_per_line_fp", default=256, help="weight_bram_width/16")
    parser.add_argument("--softmax_m", type=int, default=0)
    parser.add_argument("--hardware_group_size", type=int, default=1024, help="硬件的fpw, fpb buffer一次最多存多少数据")

    args = parser.parse_args()
    args.pth_prefix = args.pth_pre + f"{args.layer_num}."
    next_layer_num = args.layer_num + 1
    args.pth_next_prefix = args.pth_pre + f"{next_layer_num}."
    args.data_dir_fc = args.data_dir + f"layer{args.layer_num}/fc/"
    args.data_dir_attn = args.data_dir + f"layer{args.layer_num}/attention/"
    return args

args = get_args() 