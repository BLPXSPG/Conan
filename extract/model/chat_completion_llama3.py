# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM


from accelerate.utils import is_xpu_available

def load_llama_3(model_name, quantization=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16

    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
    return model,tokenizer

def llama3_predict(
    model,
        tokenizer,
        prompt,
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    dialogs = [
        [{"role": "user", "content": prompt}],
    ]
    chats = tokenizer.apply_chat_template(dialogs)

    with torch.no_grad():
        for idx, chat in enumerate(chats):

            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            if is_xpu_available():
                tokens= tokens.to("xpu:0")
            else:
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
            output_text_filter = output_text.split("assistant\n\n")[-1].strip()

            # print(output_text_filter)
            return output_text_filter
# if __name__ == "__main__":
    # main("/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/0cac6d727e4cdf117e1bde11e4c7badd8b963919","how old are you")
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/0cac6d727e4cdf117e1bde11e4c7badd8b963919")
    # dialogs = [
    #     [{"role": "user", "content": "how old are you"}],
    # ]
    # print(tokenizer.apply_chat_template(dialogs, tokenize=False))
    # print(tokenizer.apply_chat_template(dialogs))
    #
