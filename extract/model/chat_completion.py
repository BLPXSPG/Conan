# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys

import torch
from transformers import LlamaTokenizer
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig

from llama_recipes.inference.chat_utils import read_dialogs_from_file, format_tokens
from llama_recipes.inference.model_utils import load_model, load_peft_model
# from llama_recipes.inference.safety_utils import get_safety_checker

def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    return model

def llama_predict(
    model,
    prompt,
    max_new_tokens = 1000, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.75, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    **kwargs
):


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    tokenizer = LlamaTokenizer.from_pretrained('/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496')
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )
    dialogs = [
        [{"role": "user", "content": prompt}],
        ]
    chats = format_tokens(dialogs, tokenizer)

    with torch.no_grad():

        tokens= torch.tensor(chats[0]).long()
        tokens= tokens.unsqueeze(0)
        tokens= tokens.to("cuda:0")
        outputs = model.generate(
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


        # print(f"Model output:\n{output_text}")
        output = output_text.split('[/INST]')[-1]
        return output




if __name__ == "__main__":
    model_name = '/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496',
    model = load_model(model_name)
    llama_predict(model)

