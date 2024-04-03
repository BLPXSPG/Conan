## Usage

For the llama model, run the following command:

1. Extract character relationship graph from a single person's perspective.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type one_step_for_one --max_gene_len 1200
```
2. Extract character relationship graph from all characters' perspectives.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type one_step_for_all --max_gene_len 1200
```
3. Extract the character relationship graph from a single person's perspective in two steps: 
4. 
    1. Extract all characters
    2. Perform a Cartesian product to obtain pairwise characters, and inquire about the character relationships.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_pairwise_for_one --max_gene_len 800
```
4. Extract the character relationship graph from all characters' perspectives in two steps: 

   1. Extract all characters
   2. Perform a Cartesian product to obtain pairwise characters, and inquire about the character relationships.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_pairwise_for_all --max_gene_len 800
```
5. Extract all characters(Run this command before running the 6 and 7)

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type extract_character --max_gene_len 800
```

6. Extract the character relationship graph from a single person's perspective in two steps: 

   1. Extract all characters
   2. Extract the character relationship graph based on the relationships extracted.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_for_one --max_gene_len 1200
```
7. Extract the character relationship graph from all characters' perspectives in two steps: 
    1. Extract all characters
    2. Extract the character relationship graph based on the relationships extracted.
```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_for_all --max_gene_len 1200
```
8. Given the correct list of characters, directly extract all characters from a single person's perspective.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_for_one_truth --max_gene_len 1200
```
9. Given the correct list of characters, directly extract all characters from all characters' perspectives.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_for_all_truth --max_gene_len 1200
```
10. Given the correct list of characters, perform a Cartesian product and inquire about the pairwise character relationships from a single person's perspective.

```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_pairwise_for_one_truth --max_gene_len 800
```
11. Given the correct list of characters, perform a Cartesian product and inquire about the pairwise character relationships from all characters' perspectives.
```
CUDA_VISIBLE_DEVICES=0,2 python extract_llama.py --model_type llama-chat-70b --extract_type two_step_pairwise_for_all_truth --max_gene_len 800
```
