import os
from llm_args import args
from llm_create_fc_data import llm_create_fc_data
from llm_create_attn_data import llm_create_attn_data
from llm_test_attn_data import llm_test_attn_data
from llm_test_fc_data import llm_test_fc_data

def main():
    os.makedirs(args.data_dir,                      exist_ok=True)
    os.makedirs(args.data_dir_fc,                   exist_ok=True)
    os.makedirs(args.data_dir_attn,                 exist_ok=True)
    os.makedirs(args.data_dir_fc+"saved_data/",     exist_ok=True)
    os.makedirs(args.data_dir_attn+"saved_data/",   exist_ok=True)
    os.makedirs(args.data_dir_fc+"test_data/",      exist_ok=True)
    os.makedirs(args.data_dir_attn+"test_data/",    exist_ok=True)

    llm_create_fc_data()
    llm_create_attn_data()
    llm_test_fc_data()
    llm_test_attn_data()
    return

if __name__ == "__main__":
    main()