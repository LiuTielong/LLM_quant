# LLM_quant
在已有.pth文件的基础上，为FPGA准备好大模型量化后的数据，以及测试数据流。
现在的有效文件都放在llm_quant这个文件夹中。
# Directory Structure
```
├── LICENSE     
├── README.md    
├── requirements.txt    
├── configs    
│   └── zero3_efficient_config.json         # config for deepspeed acceleration
├── data
│   ├── generation_metrics.py
│   ├── get_data.py                         # dataset loading and preprocessing
│   ├── passkey_retrieval
│   │   ├── create_passkey_data.py
│   │   ├── create_passkey_data.sh
│   │   └── passkey_retrieval_accuracy.py
│   └── split_pile_file.py                  # split the Pile dataset into task-specific files
├── models
│   ├── constant.py                         # a constant function model
│   ├── get_llama2
│   │   ├── convert_llama_weights_to_hf.py  # convert llama-2 weights to huggingface format
│   │   └── download_llama2.sh
│   ├── get_model.py
│   ├── gpt_j.py
│   ├── lambda_attention.py                 # efficient implementation of lambda attention
│   ├── llama.py
│   ├── model_base.py
│   └── mpt_7b.py
├── scripts
│   ├── combine_evaluate_generation.py
│   ├── combine_results.py
│   ├── eval_downstream_tasks.py            # evaluate on passkey retrieval task
│   ├── eval_generation.py                  # evaluate generation metrics
│   └── eval_ppl_deepspeed.py               # evaluate perplexity
├── utils
│   ├── arguments.py
│   └── utils.py
└── visualization
    ├── plot_nll.py
    ├── position_pca.py
    └── relative_attention_explosion.py
```
