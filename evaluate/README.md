## Evaluation Process

We follow the steps below for evaluation:

1. **Processing Groundtruth:** Since the same character may appear with different names in the script, for ease of evaluation, we merge the names in lexicographical order to generate the `label_v2` folder.

2. **Processing the Model Outputs:** Since large models often predict relations that are not in the label system, we introduced self-reflection in our paper. This involves filtering out relations not in the label system and re-predicting them by the model for correction.

   - If you wish to generate a version without correction, use `postprocess_no_self_correct.py`.

   - For correction, taking `llama3_70b` as an example:

      - First run `postprocess_with_self_correct step1 generate self-correct json.py`, which will generate all the files for relations not in the label system (`need_clean_llama3_70b.json`) and the preliminary cleaned data (`llama3_70b_v2`).

      - Next, run `python extract_llama.py --model_type your_choice --extract_type correct_relation --max_gene_len 1200 --language chinese`, which will generate the self-correcting dictionary `correct_relation_llama70b`.

      - Finally, run `postprocess_with_self_correct step2 replace self-correct json(for llama).py` to obtain the final result `llama3_70b_v3`.

3. **Evaluation:**

   - If you need to compute the final result, run `evaluate_relation_extraction.py`.

   - For scoring difficult relations, run `evaluate_difficulty_relation.py`.

   - For calculating scores for the first step of character extraction, run `evaluate_character_extraction.py`.
