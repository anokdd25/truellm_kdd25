# README

This is the repo for the paper : [TrueLLM: Efficient, Private and Explainable Distillation 
of LLMs for Few-Shots Tabular Classification](https://openreview.net/forum?id=wjPGf6kA1w&noteId=wjPGf6kA1w) 
submitted for KDD2025

## How to use this repo

We used python 3.10 for the experiments. The environment is available there:

> pip install -r env.txt


# The pipeline

## 1. Serialize the data
    > python data_serializer.py --dataset=heart 

## 2. Train the LLM and obtain the pseudodata

### BERT
    > python src/bert_infer.py --dataset=heart --numshot=4 

### GPT-3.5
    > python src/gpt_finetune.py --api_key=<your_OpenAI_api_key> --dataset=heart --numshot=4
This will create a fine-tuning job in your OpenAI account, and it will be logged into `finetune/gpt_ft.txt`. When the fine-tuning job is finished, take the fine_tuned_model parameter of the job to perform inference, e.g. `ft:gpt-3.5-turbo-0125:personal::96gYoXJr`.
    
    > python src/gpt_evaluate.py --api_key=<your_OpenAI_api_key> --model=ft:gpt-3.5-turbo-0125:personal::96gYoXJr --dataset=heart

### GPT-4
    > python src/gpt_evaluate.py --api_key=<your_OpenAI_api_key> --model=gpt-4-turbo

### Llama 2
    > python src/llama_finetune.py --dataset=heart --numshot=4
This script will output a locally fine-tuned Llama 2 7B model to `finetune/llama2/heart_numshot4/`. To perform inference with the fine-tuned model:
    
    > python src/llama_hf.py --model=finetune/llama2/heart_numshot4/ --dataset=heart --numshot=4

### Mistral
    > python src/mistral_infer.py --api_key=<your_Mistral_api_key> --dataset=heart --numshot=4

### TabLLM
Results are obtained from the [TabLLM](https://github.com/clinicalml/TabLLM) repository and stored in `eval_res/tabllm` for both private and non-private settings.

### TabPFN
    > python src/tabpfn_eval.py --dataset=heart --numshot=4

Put --private argument to True for private setting.

## 3. Distill the LLM
    > python baseline_general.py
    
    The parameters in `__main__` are the following:
   1. `explain`: to get the rules from the explainer
   2. `verbose`: to print the AUC at each step/seed
   3. `grid`: to perform a grid search for `xgboost`
   4. `models_to_train`: to choose which models to train, by default all available models
   5. `datasets`: which datasets to work on, by default all available datasets
   6. `llm`: 'tabpfn' or 'tabllm' , the llm to distill
   7. `llm_shots`: [4, 16, 64, 256] # [0, 4, 16, 64, 256] , to direct the distillation process to the directory of llm with $n$ number of shots 
   8. `privacy`: [True] or [False], to distill from the synthetic data or not
   9. `num_shots`: [None] or [4, 16, 64, 256, 'all'], the number of shots of the explainer. Use [None] for distillation, otherwise input the number of shots for traditional training and inference with the explainer models
    



