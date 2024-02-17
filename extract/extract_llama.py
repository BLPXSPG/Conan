import sys
import os
import openai
from time import sleep
import json
import yaml
from config.utils.token_counter import count_string_tokens
from model.chat_completion import llama_predict
from config import Config
import torch
from transformers import LlamaForCausalLM, LlamaConfig
import re
from math import ceil
from tqdm import tqdm


import shutil
import itertools

class CharExtraction():
    def __init__(self, config, model_type, max_gene_len):
        # openai.api_key = config.openai_api_key
        self.config = config
        self.retry = True
        self.output_type = "category"
        self.config.max_tokens = 4096-max_gene_len
        self.max_gene_len = max_gene_len
        try:
            with open(os.path.join(os.getcwd(), "data", self.config.model, "record.json"), 'r') as f:
                self.record_dic = json.load(f)
                f.close()
        except:
            self.record_dic = {"split file": []}

        # Prompt Initialisation
        prompt_path = os.path.join(os.getcwd(), "prompts.yaml")
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        category_path = os.path.join(os.getcwd(), "equivalent_relation.json")
        with open(category_path, 'r') as f:
            self.equivalent_relation = json.load(f)
            # print(self.equivalent_relation)
            f.close()
        self.relation_merge_map = {}
        for relation in self.equivalent_relation:
            for mapping_relation in self.equivalent_relation[relation]:
                self.relation_merge_map[mapping_relation] = relation
        combined_list = []
        for key, values in self.equivalent_relation['equivalent relations'].items():
            combined_list.append(key)
            combined_list.extend(values)
        self.category_list = combined_list
        self.config.model = model_type
        if 'llama' in self.config.model:
            if '13' in self.config.model:
                self.llama_model = self.load_model(self.config.model_llama_13b_path)
            else:
                # self.llama_model = self.load_model(self.config.model_llama_70b_path)
                self.llama_model = self.load_model(self.config.model_llama_70b_path)
        self.json_response = '"character name": [["linked character", "relationship 1, relationship 2"], ["linked character", "relationship 1, relationship 2"]], "character name": [["linked character", "relationship 1, relationship 2"]]'
        # self.category_list = self.prompts['relation_list'].split("\n")
        # self.category_list = [item.lower().strip() for item in self.category_list]

    def load_model(self, model_name, quantization=False):
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            return_dict=True,
            load_in_8bit=quantization,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16
        )
        return model

    def extract_all(self, label_list, extract_type):
        prefix = 'english'
        label_list = os.listdir(os.path.join(os.getcwd(), prefix, "data_final"))
        label_list = sorted(label_list)

        label_list.reverse()
        if extract_type == 'correct_relation':
            self.correct_relation()


    # stories_dir = os.path.join(os.getcwd(), "data", "data_labelled")
        stories_dir = os.path.join(os.getcwd(), prefix, "data_final")
        for story_folder in tqdm(label_list, desc="Processing story folders"):
            if story_folder == ".DS_Store":
                continue
            story_dir = os.path.join(stories_dir, story_folder, "txt")

            # # 单个抽取

            if extract_type == 'one_step_for_one':
                self.extract_story(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_category",  story_folder), self.output_type)
            # # 不断update
            elif extract_type == 'one_step_for_all':
                self.extract_updatestory(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_category", story_folder), self.output_type)
            # # pair
            elif extract_type == 'two_step_pairwise_for_one':
                # self.extract_pairwise(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category", story_folder), self.output_type)
                self.extract_pairwise_given(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "relation_character", story_folder))

        # # pair for all
            elif extract_type == 'two_step_pairwise_for_all':
                # self.extract_pairwise_combinedstory(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category", story_folder), self.output_type)
                self.extract_pairwise_combinedstory_given(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "relation_character", story_folder))

            # # two step for one
            elif extract_type == 'extract_character':
                self.extract_character(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_character", story_folder), self.output_type)
            # # two step for one
            elif extract_type == 'two_step_for_one':
                self.character2story(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_two_step_category", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "relation_character", story_folder))
            # # two step for all
            elif extract_type == 'two_step_for_all':
                self.character2story_all(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_two_step_category", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "relation_character", story_folder))

            elif extract_type == 'two_step_for_one_truth':
                self.character2story(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_two_step_category_given", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "character", story_folder))
            # # two step for all
            elif extract_type == 'two_step_for_all_truth':
                self.character2story_all(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_two_step_category_given", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "character", story_folder))
            # # pair
            elif extract_type == 'two_step_pairwise_for_one_truth':
                self.extract_pairwise_given(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category_given", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "character", story_folder))
            # # pair for all
            elif extract_type == 'two_step_pairwise_for_all_truth':
                self.extract_pairwise_combinedstory_given(story_dir, os.path.join(os.getcwd(), prefix, self.config.model, "relation_pairwise_category_given", story_folder), self.output_type, os.path.join(os.getcwd(), prefix, self.config.model, "character", story_folder))
    def correct_relation(self):
        prompt = '''You are working on a relationship classification task, but the output includes some categories that are not defined in the relationships. 

The relationship list is ['same person as x (different reference)', 'same person as x (different identity)', "replaced x's identity", 'stranger to x', 'wife of x', 'concubine of x', 'husband of x', 'extramarital affair with x', 'secret lover of x', 'romantic relationships with x', 'lover of x', 'boyfriend of x', 'girlfriend of x', 'ex-romantic relationships with x', 'ex-boyfriend of x', 'ex-girlfriend of x', 'ex-wife of x', 'ex-husband of x', 'admirer of x', 'secret admirer of x', 'fondness of x', 'admired by x', 'liked by x', 'secret crush of x', 'fiance of x', 'fiancee of x', 'co-wives of x', 'father of x', 'father in law of x', 'adoptive father of x', 'future father in law of x', 'step-father of x', 'biological father of x', 'mother of x', 'mother in law of x', 'adoptive mother of x', 'future mother in law of x', 'step-mother of x', 'biological mother of x', 'child of x', 'son of x', 'son in law of x', 'future son in law of x', 'adoptive son of x', 'step-son of x', 'daughter of x', 'daughter in law of x', 'adoptive daughter of x', 'step-daughter of x', 'biological son of x', 'biological daughter of x', 'sibling of x', 'brother of x', 'half brother of x', 'adoptive brother of x', 'step-brother of x', 'older brother of x', 'younger brother of x', 'sister of x', 'half sister of x', 'adoptive sister of x', 'step-sister of x', 'older sister of x', 'younger sister of x', 'twin brother of x', 'twin sister of x', 'grandparent of x', 'grandfather of x', 'grandmother of x', 'grandchild of x', 'grandson of x', 'granddaughter of x', 'relative of x', 'sister in law of x', 'brother in law of x', 'nephew of x', 'aunt of x', 'uncle of x', 'niece of x', 'cousin of x', 'future relative of x', 'possibly family of x', 'friend of x', 'sworn brother of x', 'mentor of x', 'teacher of x', 'student of x', 'informant of x', 'information receiver from x', 'acquaintance of x', 'classmate of x', 'schoolmate of x', 'neighbour of x', 'helper of x', 'saviour of x', 'helped by x', 'seeker of help from x', 'saved by x', 'quest companion of x', 'crime partner of x', 'guest of x', 'host of x', 'perpetrator of x', 'bully of x', 'murderer of x', 'attempted perpetrator of x', 'attempted murderer of x', 'victim of x', 'killed by x', 'dislike of x', 'jealous of x', 'unsuccessful helper of x', 'suspect to x', 'disliked by x', 'jealous by x', 'subject of investigation for x', 'deceiver of x', 'thief of x', 'betrayer of x', 'spy of x', 'deceived by x', 'betrayed by x', 'suspected by x', 'suspicious of x', 'suspect of x', 'adversary of x', 'enemy of x', 'hate of x', 'hated by x', 'rebellion against x', 'rival of x', 'rival in love of x', "x's victim's family", "x's enemy's family", "perpetrator of x's family", 'in the lawsuit against x', 'manipulator of x', 'manipulated by x', 'superior of x', 'authority over x', 'employer of x', 'master of x', 'subordinate of x', 'employee of x', 'servant of x', 'guard of x', 'tour guide of x', 'minion of x', 'debtor of x', 'creditor of x', 'colleague of x', 'business partner of x', 'customer of x', 'buyer of x', 'tenant of x', 'patient of x', 'product provider of x', 'seller of x', 'landlord of x', 'service provider of x', 'lawyer of x', 'messenger of x', 'doctor of x']

Your output relation is 
{output_relation}

Please modify your output categories to the The relationship list and return them in JSON format.: 
{"your output relationship": "correct relationship "}
'''
        new_relationship = {}
        with open('./english/need_clean_llama.json', 'r') as file:
            data = json.load(file)
        n = 60  # 每组的元素数量
        split_data = [data[i:i + n] for i in range(0, len(data), n)]
        for output_list in tqdm(split_data, desc="Processing"):
            input_prompt = prompt.replace('{output_relation}', str(output_list))
            count = 0
            while count < 2:
                response =  llama_predict(self.llama_model, input_prompt, self.max_gene_len)
                repair_response = self.repair_json(response)
                print(repair_response)
                if repair_response is not None:
                    break
                count += 1
            if repair_response is not None:
                new_relationship.update(repair_response)
        with open('./english/correct_relation', 'w') as f:
            json.dump(new_relationship, f, ensure_ascii=False, indent=4)
            f.close()
    def extract_story(self, story_path, save_dir, output_type):
        """
        Input: story_path
        Returns: list of relationship graphs
        """

        print(story_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name + ".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)
            # prompt = self.prompts['extract_relationships_category_llama'].replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            prompt = self.prompts['extract_relationships_category'].replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            if count_string_tokens(prompt) < self.config.max_tokens:
                response =  llama_predict(self.llama_model, prompt, self.max_gene_len)
                print(response)
                print('-------------------------')
                with open(save_path, 'w') as f:
                    json.dump(response, f, ensure_ascii=False, indent=4)
                    f.close()
            else:
                self.record_dic["split file"].append(background_file)
                # with open(os.path.join(os.getcwd(), prefix, self.config.model, "record.json"), 'w') as f:
                #     json.dump(self.record_dic, f, ensure_ascii=False, indent=4)
                #     f.close()

                promt_len = count_string_tokens(self.prompts['extract_relationships_category'].replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                max_len = self.config.max_tokens - promt_len
                first, character_background_text = self.split_text_once(character_background_text, max_len)
                prompt = self.prompts['extract_relationships_category'].replace("{character_background_text}", first).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)

                while True:
                    # 计算当前能填充多少文本
                    promt_len = count_string_tokens(self.prompts['update_relationships_category'].replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len
                    # 如果剩下的填充不完，就分块
                    if count_string_tokens(character_background_text) >  max_len:
                        chunks = self.split_text(character_background_text, max_len)
                        prompt = self.prompts['update_relationships_category'].replace("{character_background_text}", chunks[0]).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        character_background_text = ''.join(chunks[1:])
                    else:
                        prompt = self.prompts['update_relationships_category'].replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        print(relationships_graph)
                        print('-------------------')
                        break
                with open(save_path, 'w') as f:
                    json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                    f.close()

    def character2story(self, story_path, save_dir, output_type,character_path):
        """
        Input: story_path
        Returns: list of relationship graphs
        """

        print(story_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name + ".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)

            with open(os.path.join(character_path, background_file.replace('txt','json')), 'r') as f:
                character_list = f.readlines()
                f.close()
            character_list = " ".join(character_list)
            character_list = str(eval(character_list))


            # prompt = self.prompts['extract_relationships_category_llama'].replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            prompt = self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            if count_string_tokens(prompt) < self.config.max_tokens:
                response =  llama_predict(self.llama_model, prompt, self.max_gene_len)
                print(response)
                print('-------------------------')
                with open(save_path, 'w') as f:
                    json.dump(response, f, ensure_ascii=False, indent=4)
                    f.close()
            else:
                self.record_dic["split file"].append(background_file)
                # with open(os.path.join(os.getcwd(), prefix, self.config.model, "record.json"), 'w') as f:
                #     json.dump(self.record_dic, f, ensure_ascii=False, indent=4)
                #     f.close()

                promt_len = count_string_tokens(self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                max_len = self.config.max_tokens - promt_len
                first, character_background_text = self.split_text_once(character_background_text, max_len)
                prompt = self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", first).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)

                while True:
                    promt_len = count_string_tokens(self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len
                    if count_string_tokens(character_background_text) >  max_len:
                        chunks = self.split_text(character_background_text, max_len)
                        prompt = self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", chunks[0]).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        character_background_text = ''.join(chunks[1:])
                    else:
                        prompt = self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        print(relationships_graph)
                        print('-------------------')
                        break
                with open(save_path, 'w') as f:
                    json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                    f.close()


    def split_text(self, text, max_length ):

        # Split the text into sentences
        sentences = re.split(r'(?<=[。！？.?!])', text)
        sentence_lengths = [count_string_tokens(sentence) for sentence in sentences]

        # Calculate the total number of tokens
        total_tokens = sum(sentence_lengths)

        # Determine the number of chunks and the target token count per chunk
        num_chunks = ceil(total_tokens / max_length)
        target_chunk_token_count = ceil(total_tokens / num_chunks)

        chunks = []
        current_chunk = []
        current_chunk_token_count = 0

        for sentence, length in zip(sentences, sentence_lengths):
            if current_chunk_token_count + length > target_chunk_token_count and current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_chunk_token_count = length
            else:
                current_chunk.append(sentence)
                current_chunk_token_count += length

        if current_chunk:
            if len(chunks) < num_chunks:
                chunks.append(''.join(current_chunk))
            else:
                chunks[-1] += ''.join(current_chunk)

        return chunks
    def split_text_once(self, text, max_length ):

        # Split the text into sentences
        sentences = re.split(r'(?<=[。！？.?!])', text)

        first_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = count_string_tokens(sentence)

            if current_length + sentence_length <= max_length:
                first_chunk.append(sentence)
                current_length += sentence_length
            else:
                break

        # Join the first chunk and calculate remaining text
        first_chunk_text = ''.join(first_chunk)
        remaining_text = text[len(first_chunk_text):]

        return first_chunk_text, remaining_text
    def extract_updatestory(self, story_path, save_dir, output_type):
        """
            We update the graph by giving additional character background story and ask llm to update the graph 
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_path):
            print(save_path, "exists")
            return None

        sys = ""
        relationships_graph = ""
        for background_file in tqdm(os.listdir(story_path), desc="Processing files"):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)

            if len(str(relationships_graph)) == 0:
                prompt = self.prompts['extract_relationships_category'].replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            else:
                prompt = self.prompts["update_relationships_category"].replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            if count_string_tokens(prompt) < self.config.max_tokens:
                response = llama_predict(self.llama_model, prompt, self.max_gene_len)
                print(response)
                relationships_graph = response
            else:

                if len(str(relationships_graph)) == 0:
                    # print("exceed limit", count_string_tokens(prompt))
                    promt_len = count_string_tokens(self.prompts['extract_relationships_category'].replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len
                    first, character_background_text = self.split_text_once(character_background_text, max_len)
                    prompt = self.prompts['extract_relationships_category'].replace("{character_background_text}", first).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                    relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                    print(relationships_graph)


                while True:
                    promt_len = count_string_tokens(self.prompts['update_relationships_category'].replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len

                    if count_string_tokens(character_background_text) >  max_len:
                        chunks = self.split_text(character_background_text, max_len)
                        prompt = self.prompts['update_relationships_category'].replace("{character_background_text}", chunks[0]).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        character_background_text = ''.join(chunks[1:])
                        print('########updata------------------')
                        print(relationships_graph)

                    else:
                        prompt = self.prompts['update_relationships_category'].replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        print('*******final-------------------')
                        print(relationships_graph)
                        break

            with open(save_path, 'w') as f:
                json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                f.close()
    def character2story_all(self, story_path, save_dir, output_type, character_path):
        """
            We update the graph by giving additional character background story and ask llm to update the graph
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_path):
            print(save_path, "exists")
            return None

        sys = ""
        relationships_graph = ""
        for background_file in tqdm(os.listdir(story_path), desc="Processing files"):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)

            with open(os.path.join(character_path, background_file.replace('txt','json')), 'r') as f:
                character_list = f.readlines()
                f.close()
            character_list = " ".join(character_list)
            character_list = str(eval(character_list))


            if len(str(relationships_graph)) == 0:
                prompt = self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", character_background_text).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            else:
                prompt = self.prompts["u_r_c_given_character"].replace("{character_list}", character_list).replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
            if count_string_tokens(prompt) < self.config.max_tokens:
                response = llama_predict(self.llama_model, prompt, self.max_gene_len)
                print(response)
                relationships_graph = response
            else:

                if len(str(relationships_graph)) == 0:
                    # print("exceed limit", count_string_tokens(prompt))
                    promt_len = count_string_tokens(self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len
                    first, character_background_text = self.split_text_once(character_background_text, max_len)
                    prompt = self.prompts['e_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", first).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                    relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                    print(relationships_graph)


                while True:
                    # 计算
                    promt_len = count_string_tokens(self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list)))
                    max_len = self.config.max_tokens - promt_len

                    if count_string_tokens(character_background_text) >  max_len:
                        chunks = self.split_text(character_background_text, max_len)
                        prompt = self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", chunks[0]).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        character_background_text = ''.join(chunks[1:])
                        print('########updata------------------')
                        print(relationships_graph)

                    else:
                        prompt = self.prompts['u_r_c_given_character'].replace("{character_list}", character_list).replace("{character_background_text}", character_background_text).replace("{relationships_graph}", str(relationships_graph)).replace("{character_name}", character_name).replace("{categories}", str(self.category_list))
                        relationships_graph = llama_predict(self.llama_model, prompt, self.max_gene_len)
                        print('*******final-------------------')
                        print(relationships_graph)
                        break

            with open(save_path, 'w') as f:
                json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                f.close()
    def process_response(self, string):
        try:
            response = string.strip().replace('\n','')
            pattern = r"\{.*\}"
            match = re.search(pattern, response)
            dict_str = match.group(0) if match else None
            dict_str = re.sub(r'\([^)]*\)', '', dict_str)
            extracted_dict = eval(dict_str)
            return extracted_dict['characters']
        except:
            try:
                matches = re.findall(r'\"(.*?)\"', string)
                matches = [i for i in matches if i != 'characters']
                return matches
            except:
                return []
    def extract_pairwise(self, story_path, save_dir, output_type):
        print(story_path)
        sys = ""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(story_path):
            relationships_graph = {}
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name+".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            print(save_path,'begin')
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)
            # extract characters first
            response = self.prompt_length_check(self.prompts['ext_character_list_prompt'], {"{character_name}": character_name}, self.prompts['update_character_list_prompt'], {"{character_name}": character_name}, "{character_list}", character_background_text)
            print(response)
            charactor_list = self.process_response(response)
            if len(charactor_list) > 20:
                charactor_list = charactor_list[:20]

            if charactor_list != []:
                character_pairs = self.generate_pairs(charactor_list)
                for character_pair in character_pairs:
                    prompt_whole = self.prompts['extract_relationships_category_pairwise_all']
                    fix_input_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{a}": character_pair[0], "{b}": character_pair[1]}
                    prompt_split = self.prompts['update_relationships_category_pairwise_all']
                    response = self.prompt_length_check(prompt_whole, fix_input_dic, prompt_split, fix_input_dic, "{relationships_graph}", character_background_text,output_type = 'relation')
                    try:
                        response = eval(response)
                        relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
                        print(character_pair, "++++++", response)
                    except:
                        continue
            print(relationships_graph)
            with open(save_path, 'w') as f:
                json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                f.close()
    def extract_pairwise_given(self, story_path, save_dir, output_type,character_path):
        print(story_path)
        sys = ""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(story_path):
            relationships_graph = {}
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name+".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            print(save_path,'begin')
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)

            with open(os.path.join(character_path, background_file.replace('txt','json')), 'r') as f:
                character_list = f.readlines()
                f.close()
            character_list = " ".join(character_list)
            character_list = eval(character_list)
            if 'relation_character' in character_path:
                if len(character_list)> 20:
                    character_list = character_list[:20]


            if character_list != []:
                character_pairs = self.generate_pairs(character_list)
                for character_pair in character_pairs:
                    prompt_whole = self.prompts['extract_relationships_category_pairwise_all']
                    fix_input_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{a}": character_pair[0], "{b}": character_pair[1]}
                    prompt_split = self.prompts['update_relationships_category_pairwise_all']
                    response = self.prompt_length_check(prompt_whole, fix_input_dic, prompt_split, fix_input_dic, "{relationships_graph}", character_background_text,output_type = 'relation')
                    try:
                        response = eval(response)
                        relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
                        print(character_pair, "++++++", response)
                    except:
                        continue
            print(relationships_graph)
            with open(save_path, 'w') as f:
                json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                f.close()
    def extract_character(self, story_path, save_dir, output_type):
        print(story_path)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        all_list =  []
        for file_name in os.listdir(save_dir):
            with open(os.path.join(save_dir, file_name), 'r') as f:
                character_list = f.readlines()
                f.close()
            character_list = " ".join(character_list)
            character_list = eval(character_list)
            all_list += character_list
        all_list = list(set(all_list))
        with open(os.path.join(save_dir, 'all.json'), 'w') as f:
            json.dump(all_list, f, ensure_ascii=False, indent=4)
            f.close()


    def extract_pairwise_combinedstory(self, story_path, save_dir, output_type):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_path):
            print(save_path, "exists")
            return None

        character_list = []
        relationships_graph = {}
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.readlines()
                f.close()
            character_background_text = " ".join(character_background_text)
            # extract characters first
            fix_input_dic = {"{character_name}": character_name}
            response = self.prompt_length_check(self.prompts['ext_character_list_prompt'], fix_input_dic,
                                                self.prompts['update_character_list_prompt'], fix_input_dic,
                                                "{character_list}", character_background_text, reuse_output=True,
                                                output_initial=str(character_list))
            character_list += self.process_response(response)
            character_list = [i for i in character_list if ' ' not in i]
            if len(character_list) > 20:
                break

        character_list = list(set(character_list))
        character_list = [i for i in character_list if not bool(re.search('[a-zA-Z]', i))]
        print("start pairwise relationship search")
        if len(character_list) > 20:
            character_list = character_list[:20]
        print(character_list)
        character_pairs = self.generate_pairs(character_list)


        for character_pair in character_pairs:

            for background_file in os.listdir(story_path):
                if background_file == ".DS_Store":
                    continue
                character_name = background_file[:-4]
                with open(os.path.join(story_path, background_file), 'r') as f:
                    character_background_text = f.readlines()
                    f.close()
                character_background_text = " ".join(character_background_text)


                prompt_whole = self.prompts['extract_relationships_category_pairwise_all']
                fix_input_dic = {"{character_name}": character_name, "{categories}": str(self.category_list),
                                 "{a}": character_pair[0], "{b}": character_pair[1]}
                prompt_split = self.prompts['update_relationships_category_pairwise_all']
                response = self.prompt_length_check(prompt_whole, fix_input_dic, prompt_split, fix_input_dic,
                                                    "{relationships_graph}", character_background_text,
                                                    reuse_output=True, output_initial=str(relationships_graph), output_type = 'relation')
                try:
                    response = eval(response)
                    relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
                    print(character_pair, "++++++", response)
                    break
                except:
                    continue
        print(relationships_graph)
        with open(save_path, 'w') as f:
            json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
            f.close()

    def extract_pairwise_combinedstory_given(self, story_path, save_dir, output_type,character_path):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_path):
            print(save_path, "exists")
            return None

        character_list = []
        relationships_graph = {}
        with open(os.path.join(character_path, 'all.json'), 'r') as f:
            character_list = f.readlines()
            f.close()
        character_list = " ".join(character_list)
        character_list = eval(character_list)

        if 'relation_character' in character_path:
            if len(character_list)> 25:
                character_list = character_list[:25]

        character_pairs = self.generate_pairs(character_list)

        for character_pair in character_pairs:

            for background_file in os.listdir(story_path):
                if background_file == ".DS_Store":
                    continue
                character_name = background_file[:-4]
                with open(os.path.join(story_path, background_file), 'r') as f:
                    character_background_text = f.readlines()
                    f.close()
                character_background_text = " ".join(character_background_text)


                prompt_whole = self.prompts['extract_relationships_category_pairwise_all']
                fix_input_dic = {"{character_name}": character_name, "{categories}": str(self.category_list),
                                 "{a}": character_pair[0], "{b}": character_pair[1]}
                prompt_split = self.prompts['update_relationships_category_pairwise_all']
                response = self.prompt_length_check(prompt_whole, fix_input_dic, prompt_split, fix_input_dic,
                                                    "{relationships_graph}", character_background_text,
                                                    reuse_output=True, output_initial=str(relationships_graph), output_type = 'relation')
                try:
                    response = eval(response)
                    relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
                    print(character_pair, "++++++", response)
                    break
                except:
                    continue
        print(relationships_graph)
        with open(save_path, 'w') as f:
            json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
            f.close()
    def prompt_length_check(self, prompt_whole, fix_input_dic, prompt_split, split_input_dic, output_need_to_update,
                            character_background_text, flex_input_key="{character_background_text}", reuse_output=False,
                            output_initial=None, sys="", n_chunk=2, output_type = '1'):
        prompt_fix = prompt_whole
        for key, value in fix_input_dic.items():
            prompt_fix = prompt_fix.replace(key, value)
        prompt = prompt_fix.replace(flex_input_key, character_background_text)

        if count_string_tokens(prompt) < self.config.max_tokens:
            output_need = llama_predict(self.llama_model, prompt, self.max_gene_len)
            if output_type == 'relation':
                output_need = self.extract_json(output_need)
            return output_need
        else:
            promt_len = count_string_tokens(prompt_fix)
            max_len = self.config.max_tokens - promt_len
            chunks = self.split_text(character_background_text, max_len)
            prompt_fix = prompt_whole
            for key, value in split_input_dic.items():
                prompt_fix = prompt_fix.replace(key, value)
            if reuse_output:
                prompt = prompt_fix.replace(flex_input_key, chunks[0]).replace(output_need_to_update, output_initial)
            else:
                prompt = prompt_fix.replace(flex_input_key, chunks[0])
            output_need = llama_predict(self.llama_model, prompt, self.max_gene_len)
            if output_type == 'relation':
                output_need = self.extract_json(output_need)
            print(output_need)
            if self.is_json(output_need):
                return output_need
            character_background_text = ''.join(chunks[1:])

            n = 0
            while True:
            # 计算
                prompt_fix = prompt_split
                for key, value in split_input_dic.items():
                    prompt_fix = prompt_fix.replace(key, value)
                promt_len = count_string_tokens(prompt_fix.replace(output_need_to_update, str(output_need)))
                max_len = self.config.max_tokens - promt_len

                if count_string_tokens(character_background_text) >  max_len:
                    chunks = self.split_text(character_background_text, max_len)
                    prompt = prompt_fix.replace(flex_input_key, chunks[0]).replace(output_need_to_update, str(output_need))
                    output_need = llama_predict(self.llama_model, prompt, self.max_gene_len)
                    if output_type == 'relation':
                        output_need = self.extract_json(output_need)
                    if self.is_json(output_need):
                        return output_need
                    character_background_text = ''.join(chunks[1:])
                    n += 1
                    print('########updata------------------')
                    print(output_need)
                else:
                    prompt = prompt_fix.replace(flex_input_key, character_background_text).replace(output_need_to_update, str(output_need))
                    output_need = llama_predict(self.llama_model, prompt, self.max_gene_len)
                    if output_type == 'relation':
                        output_need = self.extract_json(output_need)
                    print('*******final-------------------')
                    print(output_need)
                    return output_need

    def is_json(self,string):
        try:
            eval(string)
            return True
        except:
            return False

    def repair_json(self, extract):
        if "{" in extract:
            index_position = [i for i, c in enumerate(extract) if c == "{"]
            index_position.reverse()
        else:
            return None
        for pos in index_position:
            data = extract[pos:]
            for i in range(len(data), 0, -1):
                try:
                    character_data = data[:i]
                    if '}' not in data:
                        character_data = character_data + '}'
                    parsed = json.loads(character_data)
                    return parsed
                except json.JSONDecodeError:
                    continue

        return None

    def extract_json(self, string):
        try:
            response = string.strip().replace('\n','')
            if 'JSON format:' in response:
                response = response.split('JSON format:')[-1]
            if 'Explanation' in response:
                response = response.split('Explanation')[0]
            try:
                eval(response)
                return response
            except:
                pass

            matches = re.findall(r'\{.*?\}', response)
            last_match = matches[-1] if matches else None
            dict_str = re.sub(r'\([^)]*\)', '', last_match)
            # extracted_dict = eval(dict_str)
            return dict_str
        except:
            return string
    def merge_pair_relation_to_relationships_graph(self, pair_relation, relationships_graph):
            for key, value in pair_relation.items():
                if key in relationships_graph:
                    # Update existing entry
                    relationships_graph[key] += [value]
                else:
                    # Add new entry
                    relationships_graph[key] = [value]
            return relationships_graph

    def generate_pairs(self, input_list):
        return list(itertools.combinations(input_list, 2))

    def character_relationship(self, narrative_name: str, character_index: int, characters: list):
        character_name = characters[character_index]["Name"]
        other_characters = characters[:character_index] + characters[character_index + 1:]
        other_characters = [character["Name"] for character in other_characters]
        sys = ""
        prompt = self.prompts['relation_identification_prompt'].replace("{character_name}", character_name).replace(
            "{other_characters}", str(other_characters))
        response = self.gen_response_35(sys, prompt)
        print(sys, prompt)
        print(response)

        classes = self.prompts['relation_identification_class']
        classes = classes.split('\n')
        classes = [item.strip() for item in classes]
        output = {}
        for relationship in classes:
            output[relationship] = []
        for relationship in response:
            if (relationship['Relationship'] in classes) and (relationship['Name'] in other_characters):
                output[relationship['Relationship']] += [relationship['Name']]
            else:
                print("Unmatched relationship for %s: " % character_name, relationship)
        # output = self.check_key(classes, response, [""]*len(classes))
        print(output)
        return output

    def save_story(self, narrative_name: str, response: dict):
        file_name = narrative_name.replace(" ", "-")
        with open(os.path.join(self.save_path, file_name + ".json"), 'w') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
            f.close()

    def check_key(self, keys, check_dic, default_value):
        output = {}
        for i, key in enumerate(keys):
            if key not in check_dic:
                check_dic[key] = default_value[i]
            else:
                output[key] = check_dic[key]
        return output

    def gen_response_35(self, sys: str, prompt: str):
        try:
            deployment_id = self.config.get_azure_deployment_id_for_model(self.config.model)
            completion = openai.ChatCompletion.create(
                engine=deployment_id,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                n=1
            )
            try:
                response = completion.choices[0].message["content"].replace('`', '').replace('\n', '').replace('json',
                                                                                                               '')
                data = json.loads(response)
                return data

            except json.JSONDecodeError as e:
                if self.retry:
                    sys = sys + "Careful about comma in JSON format."
                    self.retry = False
                    print("JSONDecodeError, RETRY", response)
                    return self.gen_response_35(sys, prompt)
                else:
                    self.retry = True
                    print("JSONDecodeError: ", e)
                    print("Couldn't fix the JSON", response)
                    return {}

        except Exception as e:
            print("ERROR response", e)
            sleep(5)
            if self.retry:
                self.retry = False
                return self.gen_response_35(sys, prompt)
            else:
                self.retry = True
                return {}


if __name__ == "__main__":
    import argparse
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Character Extraction Script")
    parser.add_argument("--model_type", type=str, help="Description for param1")
    parser.add_argument("--extract_type", type=str, help="Description for param2")
    parser.add_argument("--max_gene_len", type=int, help="Description for param2")
    #['one_step_for_one','one_step_for_all','two_step_pairwise_for_one','two_step_pairwise_for_all']
    # 解析命令行参数
    args = parser.parse_args()

    label_list = ['002-奕剑决（8人开放）',
                  '003-死穿白（8人开放）',
                  '005-幽灵复仇（7人)',
                  '007-雪地惊情（10~12开放）',
                  '033-丹水山庄（7人）',
                  '053-未完结的爱（7人开放）',
                  '054-东方之星号游轮事件（5人开放）',
                  '060-魔盗之殇（5人开放）',
                  '076-孝衣新娘（10人）',
                  '131-罪恶（4人封闭）',
                  '134-致命喷泉（4人封闭）',
                  '139-血海无涯（5人封闭）',
                  '141-校园不思议事件（5人）',
                  '142-西门变（5人封闭）',
                  '152-绝命阳光号（4人封闭）',
                  '160-病毒（4人封闭）',
                  '191-江湖客栈（4人封闭）',
                  '246-紫藤夫人（6人）',
                  # '252-梦山（6人封闭）',
                  '435-鄂西山灵（6人）',
                  '534-作揖（7人封闭-古风烧脑）',
                  '655-章东镇迷案（6人）',
                  '784-失真的旋律（6人）',
                  # '1444-枉痴心（6人）',
                  '1702-孤舟萤（6人）',
                  '1849-曼娜（6人）']
    label_list = ['002-Yijianjue (open to 8 people)',
                  '003-Death wears white (open to 8 people)',
                  '005-Ghost Revenge (7 people)',
                  '007-Snowy scene (open from 10 to 12)',
                  '033-Danshui Villa (7 people)',
                  '053-Unfinished love (open to 7 people)',
                  '054-Oriental Star Cruise Incident (open to 5 people)',
                  '060-The Sorrow of the Demon Thief (open to 5 people)',
                  '076-Xiaoyi Bride (10 people)',
                  '131-Sin (4 people closed)',
                  '134 - Deadly Fountain (4 people closed)',
                  '139-Boundless sea of \u200b\u200bblood (5 people closed)',
                  '141-Unbelievable incident on campus (5 people)',
                  '142-Ximen Bian (5 people closed)',
                  '152-Desperate Sunshine (4 people closed)',
                  '160-Virus (4 people closed)',
                  '191-Jianghu Inn (4 people closed)',
                  '246-Mrs. Wisteria (6 people)',
                  '435-Western Hubei Shanling (6 people)',
                  '534 - Bowing (7 people closed - ancient style brain-burning)',
                  '655-The Mysterious Case of Zhangdong Town (6 people)',
                  '784-Distorted Melody (6 people)',
                  '1702-Guzhouying (6 people)',
                  '1849-Manna (6 people)']


    cfg = Config()
    ext = CharExtraction(cfg, args.model_type, args.max_gene_len)

    ext.extract_all(label_list, args.extract_type)

