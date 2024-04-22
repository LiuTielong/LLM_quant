# LLM_quant
在已有.pth文件的基础上，为FPGA准备好大模型量化后的数据，以及测试数据流。
现在的有效文件都放在llm_quant这个文件夹中。
# Directory Structure
```
├── LICENSE     
├── README.md
├── llm_quant
│   ├──main.py
│   ├──llm_args.py
│   ├──llm_create_attn_data.py
│   ├──llm_create_fc_data.py
│   ├──llm_test_attn_data.py
│   ├──llm_test_fc_data.py
│   ├──llm_test_attn_data_simple.py
│   ├──llm_test_fc_data_simple.py
│   ├──llm_validate_attn_data.py
│   └──transfer_data.py
```
