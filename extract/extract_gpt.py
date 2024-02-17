import os
import openai 
from time import sleep
import json
import yaml
#from memory.processing import split_text
from config.utils.token_counter import count_string_tokens
import shutil
from googletrans import Translator
from deep_translator import GoogleTranslator
import itertools
import re
from math import ceil

class CharExtraction():
    def __init__(self, config):
        self.config = config
        self.retry = True
        self.output_type = "category"

        # Prompt Initialisation
        prompt_path = os.path.join(os.getcwd(), "prompts.yaml")
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        
        category_path = os.path.join(os.getcwd(), "data", "equivalent_relation.json")
        with open(category_path, 'r') as f:
            self.equivalent_relation = json.load(f)
            f.close()
        self.high_level_relations = self.equivalent_relation["High-Level relations"]
        self.high_level_relations = {k.lower().strip(): v for k, v in self.high_level_relations.items()}

        self.high_level_category_map = {}
        for relation in self.high_level_relations:
            for mapping_relation in self.high_level_relations[relation]:
                relation = relation.lower().strip()
                mapping_relation = mapping_relation.lower().strip()
                self.high_level_category_map[mapping_relation] = relation
        self.equivalent_relation = self.equivalent_relation["equivalent relations"]
        print(len(list(self.equivalent_relation.keys())))
        self.equivalent_relation = {k.lower().strip(): v for k, v in self.equivalent_relation.items()}
        self.relation_merge_map = {}
        for relation in self.equivalent_relation:
            for mapping_relation in self.equivalent_relation[relation]:
                relation = relation.lower().strip()
                mapping_relation = mapping_relation.lower().strip()
                self.relation_merge_map[mapping_relation] = relation
        
        self.category_list = list(self.equivalent_relation.keys()) + list(self.relation_merge_map.keys())
        self.category_list = [item.lower().strip() for item in self.category_list]
        self.category_list_n = [item.replace("-"," ") for item in self.category_list]
        print(len(self.category_list_n))

    def extract_all(self, language="chinese"):
        """
        For all script we saved, extract the relationships and merge the graph
        """
        annotation_path = os.path.join(os.getcwd(), "data", language, "label")
        stories_dir = os.path.join(os.getcwd(), "data", language, "data_final")
        for story_folder in os.listdir(annotation_path):
        #for story_folder in os.listdir(stories_dir):
            if story_folder == ".DS_Store":
                continue
            story_dir = os.path.join(stories_dir, story_folder, "txt")
            self.annotation_dir = os.path.join(annotation_path, story_folder)
            character_dir = os.path.join(os.getcwd(), "data", language, self.config.model, "character", story_folder)

            # extract the characters and relations all together
            save_dir = os.path.join(os.getcwd(), "data", language, self.config.model, "extract_whole_graph", story_folder)
            self.extract_individual_character_relations(story_dir, "", save_dir, character="none")
            self.extract_all_character_relations(story_dir, "", save_dir, character="none")
            # extract characters
            self.extract_character(story_dir, character_dir)
            # given characters (can be extracted characters or labelled characters), extract relations directly
            save_dir = os.path.join(os.getcwd(), "data", language, self.config.model, "relation_extract_directly", story_folder)
            self.extract_individual_character_relations(story_dir, character_dir, os.path.join(save_dir, "given"), character="given")
            self.extract_all_character_relations(story_dir, character_dir, os.path.join(save_dir, "given"), character="given")
            self.extract_individual_character_relations(story_dir, character_dir, os.path.join(save_dir, "extract"), character="extract")
            self.extract_all_character_relations(story_dir, character_dir, os.path.join(save_dir, "extract"), character="extract")
            # given characters (can be extracted characters or labelled characters), get relations pairwisely
            save_dir = os.path.join(os.getcwd(), "data", language, self.config.model, "relation_extract_pairwisely", story_folder)
            self.extract_relations_pairwisely(story_dir, character_dir, os.path.join(save_dir, "given"), character="given")
            self.extract_relations_pairwisely(story_dir, character_dir, os.path.join(save_dir, "extract"), character="extract")

    def out_of_limit(self, prompt_format, update_prompt_format, fixed_replace_dic, variable_key, variable_value, character_background_text, save_path, sys="", ifsave=True, error_save={}):
        """
        prompt_format: e.g. 'extract_relation_given_character'
        update_prompt_format: e.g. 'update_relationships_category'
        fixed_replace_dic: e.g. {"{character_name}": character_name, "{categories}": str(self.category_list)}
        variable_key: e.g. "{relationships_graph}"
        variable_value: e.g. relationships_graph
        background_text: "the story ... "
        """
        prompt = self.prompts[prompt_format]
        for key, value in fixed_replace_dic.items():
            prompt = prompt.replace(key, str(value))
        prompt_current = prompt.replace(variable_key, str(variable_value))
        prompt_current= prompt_current.replace("{character_background_text}", character_background_text)

        if count_string_tokens(prompt_current) < self.config.max_tokens:
            #print("pass")
            response = self.gen_response_35(sys, prompt_current)
            if len(list(response.keys())) > 0:
                print(response)
                if ifsave:
                    with open(save_path, 'w') as f:
                        json.dump(response, f, ensure_ascii=False, indent=4)
                        f.close()
            else:
                print("error, save empty file")
                if ifsave:
                    with open(save_path, 'w') as f:
                        json.dump(error_save, f, ensure_ascii=False, indent=4)
                        f.close()
            return response
        else:
            #print("split")
            if len(list(variable_value.keys())) == 0:
                prompt_current = prompt.replace(variable_key, str(variable_value))
                promt_len = count_string_tokens(prompt_current)
                max_len = self.config.max_tokens - promt_len
                first, character_background_text = self.split_text_once(character_background_text, max_len)
                
                prompt = prompt_current.replace("{character_background_text}", first)
                variable_value = self.gen_response_35(sys, prompt)

            prompt = self.prompts[update_prompt_format]
            for key, value in fixed_replace_dic.items():
                prompt = prompt.replace(key, str(value))
            while True:
                prompt_current = prompt.replace(variable_key, str(variable_value))
                promt_len = count_string_tokens(prompt_current)
                max_len = self.config.max_tokens - promt_len
                if count_string_tokens(character_background_text) >  max_len:
                    chunks = self.split_text(character_background_text, max_len)
                    prompt_current = prompt.replace("{character_background_text}", chunks[0]).replace(variable_key, str(variable_value))
                    response = self.gen_response_35(sys, prompt_current)
                    if len(list(response.keys())) > 0:
                        variable_value = response
                    character_background_text = ''.join(chunks[1:])
                else:
                    prompt_current = prompt.replace("{character_background_text}", character_background_text).replace(variable_key, str(variable_value))
                    response = self.gen_response_35(sys, prompt_current)
                    if len(list(response.keys())) > 0:
                        variable_value = response
                    break
            if ifsave:
                with open(save_path, 'w') as f:
                    json.dump(variable_value, f, ensure_ascii=False, indent=4)
                    f.close()
            return variable_value
            
    def extract_individual_character_relations(self, story_path, character_dir, save_dir, character="none"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name+".json")
            print(save_path)
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.read()
                f.close()
            if character == "extract":
                with open(os.path.join(character_dir, character_name+".json"), 'r') as f:
                    character_list = json.load(f)["characters"]
                    f.close()
                fixed_replace_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{character_list}": character_list}
                self.out_of_limit('extract_relation_given_character', 'update_relation_given_character', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path)
            elif character == "given":
                with open(os.path.join(self.annotation_dir, character_name+".json"), "r") as f:
                    character_list = list(json.load(f).keys())
                    f.close()
                fixed_replace_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{character_list}": character_list}
                self.out_of_limit('extract_relation_given_character', 'update_relation_given_character', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path)
            elif character == "none":
                fixed_replace_dic = {"{character_name}": character_name, "{categories}": str(self.category_list)}
                self.out_of_limit('extract_relation_and_character', 'update_relation_and_character', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path)

    def extract_all_character_relations(self, story_path, character_dir, save_dir, character="none"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_path):
            print(save_path, "exists")
            return None

        if character == "extract":
            with open(os.path.join(character_dir, "all.json"), 'r') as f:
                character_list = json.load(f)
                f.close()
                if "characters" in character_list:
                    character_list = character_list["characters"]
                else:
                    with open(save_path, 'w') as f:
                        json.dump({}, f, ensure_ascii=False, indent=4)
                        f.close()
        elif character == "given":
            with open(os.path.join(self.annotation_dir, "all.json"), "r") as f:
                character_list = list(json.load(f).keys())
                f.close()
        
        text_all = ""
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.read()
                f.close()
            text_all += self.prompts["extract_relationships_individual"].replace("{character_background_text}",character_background_text).replace("{character_name}", character_name)
        
        if (character == "extract") or (character == "given"):
            fixed_replace_dic = {"{categories}": str(self.category_list), "{character_list}": character_list}
            relationships_graph = self.out_of_limit('extract_relation_given_character_all', 'update_relation_given_character_all', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path)
        elif character == "none":
            fixed_replace_dic = {"{categories}": str(self.category_list)}
            relationships_graph = self.out_of_limit('extract_relation_and_character_all', 'update_relation_and_character_all', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path)
            
    def extract_character(self, story_path, character_dir):
        if not os.path.isdir(character_dir):
            os.makedirs(character_dir)
        character_all = {}
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(character_dir, character_name+".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.read()
                f.close()
            fixed_replace_dic = {"{character_name}": character_name}
            self.out_of_limit('ext_character_list_prompt', 'update_character_list_prompt', fixed_replace_dic, "{character_list}", {}, character_background_text, save_path, error_save={"characters":[]})

        save_all_path = os.path.join(character_dir, "all.json")
        character_list_all = []
        for character_file in os.listdir(character_dir):
            if (character_file == ".DS_Store") or (character_file == "all.json"):
                continue
            with open(os.path.join(character_dir, character_file), 'r') as f:
                character_list = json.load(f)["characters"]
                f.close()
            character_list_all += character_list
        character_list_all = list(set(character_list_all))
        with open(save_all_path, 'w') as f:
            json.dump({"characters":character_list_all}, f, ensure_ascii=False, indent=4)
            f.close()

    def extract_relations_pairwisely(self, story_path, character_dir, save_dir, character="extract"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            save_path = os.path.join(save_dir, character_name+".json")
            if os.path.exists(save_path):
                print(save_path, "exists")
                continue
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.read()
                f.close()
            if character == "extract":
                with open(os.path.join(character_dir, character_name+".json"), 'r') as f:
                    character_list = json.load(f)["characters"]
                    f.close()
            elif character == "given":
               with open(os.path.join(self.annotation_dir, character_name+".json"), "r") as f:
                    character_list = list(json.load(f).keys())
                    f.close()
            character_pairs = self.generate_pairs(character_list)
            relationships_graph = {}
            for character_pair in character_pairs:
                fixed_replace_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{character_list}": character_list, "{a}": character_pair[0], "{b}": character_pair[1]}
                response = self.out_of_limit('extract_relationships_category_pairwise', 'update_relationships_category_pairwise', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path, ifsave=False)
                relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
                #print(character_pair, "++++++", relationships_graph)
            with open(save_path, 'w') as f:
                json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
                f.close()

        #relationships_graph_all = {}
        save_all_path = os.path.join(save_dir, "all.json")
        if os.path.exists(save_all_path):
            print(save_all_path, "exists")
            return None
        text_all = ""
        for background_file in os.listdir(story_path):
            if background_file == ".DS_Store":
                continue
            character_name = background_file[:-4]
            with open(os.path.join(story_path, background_file), 'r') as f:
                character_background_text = f.read()
                f.close()
            text_all += self.prompts["extract_relationships_individual"].replace("{character_background_text}",character_background_text).replace("{character_name}", character_name)
        
        if character == "extract":
            with open(os.path.join(character_dir, "all.json"), 'r') as f:
                character_list = json.load(f)["characters"]
                f.close()
        elif character == "given":
            with open(os.path.join(self.annotation_dir, "all.json"), "r") as f:
                character_list = list(json.load(f).keys())
                f.close()
        character_pairs = self.generate_pairs(character_list)
        relationships_graph = {}
        for character_pair in character_pairs:
            fixed_replace_dic = {"{character_name}": character_name, "{categories}": str(self.category_list), "{character_list}": character_list, "{a}": character_pair[0], "{b}": character_pair[1]}
            response = self.out_of_limit('extract_relationships_category_pairwise', 'update_relationships_category_pairwise', fixed_replace_dic, "{relationships_graph}", {}, character_background_text, save_path, ifsave=False)
            relationships_graph = self.merge_pair_relation_to_relationships_graph(response, relationships_graph)
        #relationships_graph_all = self.merge_pair_relation_to_relationships_graph_all(relationships_graph, relationships_graph_all)
        print("*******", relationships_graph)
        with open(save_all_path, 'w') as f:
            json.dump(relationships_graph, f, ensure_ascii=False, indent=4)
            f.close()
    
    def split_text(self, text, max_length):

        # Split the text into sentences
        sentences = re.split(r'(?<=[。！？])', text)
        sentence_lengths = [count_string_tokens(sentence) for sentence in sentences]

        # Calculate the total number of tokens
        total_tokens = sum(sentence_lengths)

        # Determine the number of chunks and the target token count per chunk
        num_chunks = ceil(total_tokens / max_length)
        if num_chunks == 0:
            print(total_tokens, max_length)
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
    
    def split_text_once(self, text, max_length):

        # Split the text into sentences
        sentences = re.split(r'(?<=[。！？])', text)

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

    def merge_pair_relation_to_relationships_graph(self, pair_relation, relationships_graph):
        #print(pair_relation, relationships_graph)
        try:
            for key, value in pair_relation.items():
                #print("key, value", key, value)
                if key in relationships_graph:
                    # Update existing entry
                    relationships_graph[key] += [value]
                else:
                    # Add new entry
                    relationships_graph[key] = [value]
            return relationships_graph
        except AttributeError:
            return relationships_graph

    def merge_pair_relation_to_relationships_graph_all(self, relationships_graph, relationships_graph_all):
        print(relationships_graph_all)
        for key, value in relationships_graph.items():
            print("key, value", key, value)
            if key in relationships_graph_all:
                # Update existing entry
                relationships_graph_all[key] += value
            else:
                # Add new entry
                relationships_graph_all[key] = value
        return relationships_graph_all
    
    def generate_pairs(self, input_list):
        return list(itertools.combinations(input_list, 2))

    def save_story(self, narrative_name: str, response: dict):
        file_name = narrative_name.replace(" ", "-")
        with open(os.path.join(self.save_path, file_name+".json"), 'w') as f:
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
    
    def check_relation_language(self, language="chinese"):
        stats_dir = os.path.join(os.getcwd(), "data", "translation")
        if not os.path.isdir(stats_dir):
            os.makedirs(stats_dir)
        gpt4_relationship_to_specified_path = os.path.join(stats_dir, self.config.model + "_relationship_to_specified.json")
        if os.path.exists(gpt4_relationship_to_specified_path):
            with open(gpt4_relationship_to_specified_path, "r") as f:
                relationship_to_specified = json.load(f)
                f.close()
        else:
            relationship_to_specified = {}

        # check all extracted relationships and turn incorrect one into correct categories
        annotation_path = os.path.join(os.getcwd(), "data", language, "label")
        stories_dir = os.path.join(os.getcwd(), "data", language, "extract_whole_graph")
        for story_folder in os.listdir(annotation_path):
            if story_folder == ".DS_Store":
                continue
            # extract the characters and relations all together
            extract_whole_graph_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v2", "extract_whole_graph", story_folder)
            extract_whole_graph_save_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v3", "extract_whole_graph", story_folder)
            _, relationship_to_specified = self.check_one_story(extract_whole_graph_dir, extract_whole_graph_save_dir, gpt4_relationship_to_specified_path, relationship_to_specified)

            # given characters (can be extracted characters or labelled characters), extract relations directly
            relation_extract_directly_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v2", "relation_extract_directly", story_folder)
            relation_extract_directly_save_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v3", "relation_extract_directly", story_folder)
            _, relationship_to_specified = self.check_one_story(os.path.join(relation_extract_directly_dir, "given"), os.path.join(relation_extract_directly_save_dir, "given"), gpt4_relationship_to_specified_path, relationship_to_specified)
            _, relationship_to_specified = self.check_one_story(os.path.join(relation_extract_directly_dir, "extract"), os.path.join(relation_extract_directly_save_dir, "extract"), gpt4_relationship_to_specified_path, relationship_to_specified)

            # given characters (can be extracted characters or labelled characters), get relations pairwisely
            relation_extract_pairwisely_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v2", "relation_extract_pairwisely", story_folder)
            relation_extract_pairwisely_save_dir = os.path.join(os.getcwd(), "data", language, self.config.model + "_v3", "relation_extract_pairwisely", story_folder)
            _, relationship_to_specified = self.check_one_story(os.path.join(relation_extract_pairwisely_dir, "given"), os.path.join(relation_extract_pairwisely_save_dir, "given"), gpt4_relationship_to_specified_path, relationship_to_specified)
            _, relationship_to_specified = self.check_one_story(os.path.join(relation_extract_pairwisely_dir, "extract"), os.path.join(relation_extract_pairwisely_save_dir, "extract"), gpt4_relationship_to_specified_path, relationship_to_specified)

    def check_one_story(self, load_dir, save_dir, gpt4_relationship_to_specified_path, relationship_to_specified):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for background_file in os.listdir(load_dir):
            if background_file == ".DS_Store":
                continue
            load_path = os.path.join(load_dir, background_file)
            save_path = os.path.join(save_dir, background_file)
            with open(load_path, "r") as f:
                relationships_dic = json.load(f)
                f.close()
            relationships_dic, relationship_to_specified = self.relationships_to_english(relationships_dic, relationship_to_specified, save_path, gpt4_relationship_to_specified_path)
        return relationships_dic, relationship_to_specified
    
    def relationships_to_english(self, relationships_dic, relationship_to_specified, save_path, gpt4_relationship_to_specified_path, ood = False):
        relationships_ood_all = []
        for character, relationships in relationships_dic.items():
            for i, relationship in enumerate(relationships):
                relationship_list = relationship[1].split(",")
                relationships_ood = [r.lower().strip() for r in relationship_list if r.lower().strip().replace("-"," ") not in self.category_list_n]
                relationships_ood_all += relationships_ood

        if len(relationships_ood_all) > 0:
            ood = True
            relationships_ood_all = list(set(relationships_ood_all))
            #relationship_list_unknown = [r for r in relationships_ood if r not in relationship_to_specified]
            prompt = self.prompts["re_category"].replace("{relationship_list}", str(relationships_ood_all)).replace("{categories}", str(self.category_list))
            updated_dic = self.gen_response_35("", prompt)
            new_dict = {key: value for key, value in updated_dic.items() if value in self.category_list_n}
            print("update_dic", new_dict)
            new_dict.update(relationship_to_specified)
            relationship_to_specified = new_dict.copy()

        for character, relationships in relationships_dic.items():
            for i, relationship in enumerate(relationships):
                if ood:
                    relationship_list = relationship[1].split(",")
                    relationship_list = [r.lower().strip() for r in relationship_list]
                    relationship_list_known = [relationship_to_specified[r] for r in relationship_list if r in relationship_to_specified]
                    relationship_list_unknown = [r for r in relationship_list if r not in relationship_to_specified]
                    relationships[i] = [relationship[0], ", ".join(relationship_list_known + relationship_list_unknown)]
                    print(relationship_list_unknown)

            relationships_dic[character] = relationships
        with open(save_path, 'w') as f:
            json.dump(relationships_dic, f, ensure_ascii=False, indent=4)
            f.close()
        if ood:
            with open(gpt4_relationship_to_specified_path, 'w') as f:
                json.dump(relationship_to_specified, f, ensure_ascii=False, indent=4)
                f.close()
        return relationships_dic, relationship_to_specified

    def is_chinese(s):
        for char in s:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def translate_all(self, language="chinese"):
        stats_dir = os.path.join(os.getcwd(), "data", "translation")
        if not os.path.isdir(stats_dir):
            os.makedirs(stats_dir)

        # first translate labelled character names
        label_dir = os.path.join(os.getcwd(), "data", language, "label")
        char_save_path = os.path.join(stats_dir, "char_name_list.json")
        name_to_english_dir = os.path.join(stats_dir, "name_to_english.json")
        name_list_dic = self.get_name_list(label_dir, char_save_path)
        name_to_english_dic = self.name_to_english(name_list_dic, name_to_english_dir)

        extract_name_to_english_dic = {}

        #translate story names
        stories_dir = os.path.join(os.getcwd(), "data", language, "data_final")
        story_to_english_dir = os.path.join(os.getcwd(), "data", "story_to_english.json")
        story_to_english_dic = self.story_to_english(stories_dir, story_to_english_dir)

        #translate input stories
        engligh_version_dir = os.path.join(os.getcwd(), "data", "english", "data_final")
        self.translate_input(stories_dir, engligh_version_dir, name_to_english_dic, extract_name_to_english_dic, story_to_english_dic)
        #translate labels
        engligh_version_dir = os.path.join(os.getcwd(), "data", "english", "label")
        self.translate_label(label_dir, engligh_version_dir, name_to_english_dic, extract_name_to_english_dic, story_to_english_dic)

    def get_name_list(self, stories_dir, char_save_path):
        if os.path.exists(char_save_path):
            with open(char_save_path, "r") as f:
                name_list_dic = json.load(f)
                f.close()
        else:
            name_list_dic = {}
            for story_folder in os.listdir(stories_dir):
                if story_folder == ".DS_Store":
                    continue
                source_dir = os.path.join(stories_dir, story_folder)
                character_list = []
                for background_file in os.listdir(source_dir):
                    if story_folder == ".DS_Store":
                        continue
                    character_name, ext = os.path.splitext(background_file)
                    with open(os.path.join(source_dir, background_file), "r") as f:
                        relationships_dic = json.load(f)
                        f.close()
                    character_list += list(relationships_dic.keys()) 
                    if character_name != "all":
                        character_list += [character_name]
                    for key, value in relationships_dic.items():
                        character_list += [char[0] for char in value]
                character_list = list(set(character_list))
                name_list_dic[story_folder] = character_list
                print(character_list)
            with open(char_save_path, 'w') as f:
                json.dump(name_list_dic, f, ensure_ascii=False, indent=4)
                f.close()
        return name_list_dic

    def name_to_english(self, name_list_dic, save_path):
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                name_to_english_dic = json.load(f)
                f.close()
        else:
            name_to_english_dic = {}
        for story_folder, character_list in name_list_dic.items():
            # check if there exists untranslated characters
            if story_folder in name_to_english_dic:
                translated_characters = name_to_english_dic[story_folder]
                untranslated_character_list = [character for character in character_list if character not in translated_characters]
                if len(untranslated_character_list) > 0:
                    print(story_folder, "untranslated character:", untranslated_character_list)
                    prompt = self.prompts["trasnlate_names"].replace("{character_names}", str(untranslated_character_list))
                    translated_dic = self.gen_response_35("", prompt)
                    translated_dic.update(translated_characters)
                else:
                    translated_dic = translated_characters
            else:
                prompt = self.prompts["trasnlate_names"].replace("{character_names}", str(character_list))
                translated_dic = self.gen_response_35("", prompt)
            name_to_english_dic[story_folder] = translated_dic
            print(character_list, translated_dic)
        with open(save_path, 'w') as f:
            json.dump(name_to_english_dic, f, ensure_ascii=False, indent=4)
            f.close()
        return name_to_english_dic

    def story_to_english(self, stories_dir, save_dir):
        if os.path.exists(save_dir):
            with open(save_dir, "r") as f:
                story_to_english_dic = json.load(f)
                f.close()
        else:
            story_to_english_dic = {}
            translator = GoogleTranslator(source='auto', target='en')
            for story_folder in os.listdir(stories_dir):
                if story_folder == ".DS_Store":
                    continue
                translated_story_folder = translator.translate(story_folder)
                story_to_english_dic[story_folder] = translated_story_folder
            with open(save_dir, 'w') as f:
                json.dump(story_to_english_dic, f, ensure_ascii=False, indent=4)
                f.close()
        print(story_to_english_dic)
        return story_to_english_dic
    
    def translate_input(self, stories_dir, engligh_version_dir, name_to_english_dic, extract_name_to_english_dic, story_to_english_dic):
        #translator = GoogleTranslator(source='auto', target='en')
        story_total = list(name_to_english_dic.keys()) +  list(extract_name_to_english_dic.keys())
        for story_folder in story_total:
            if story_folder in name_to_english_dic:
                name_to_english_this_story = name_to_english_dic[story_folder]
                print("label", story_folder, name_to_english_this_story)
            else:
                name_to_english_this_story = extract_name_to_english_dic[story_folder]
                print("extract", story_folder, name_to_english_this_story)
            #translated_story_folder = translator.translate(story_folder)
            translated_story_folder = story_to_english_dic[story_folder]
            target_dir = os.path.join(engligh_version_dir, translated_story_folder, "txt")
            if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
            source_dir = os.path.join(stories_dir, story_folder, "txt")
            for background_file in os.listdir(source_dir):
                if background_file == ".DS_Store":
                    continue
                else:
                    name, ext = os.path.splitext(background_file)
                    target_path = os.path.join(target_dir, name_to_english_this_story[name] + ext)
                    if os.path.isfile(target_path):
                        continue
                    else:
                        print(target_path)
                        with open(os.path.join(source_dir, background_file), 'r') as f:
                            character_background_text = f.read()
                            f.close()
                        for chinesename, englishname in name_to_english_this_story.items():
                            character_background_text = character_background_text.replace(chinesename, englishname)
                        translated_content = self.translate_text(character_background_text)
                        if translated_content:
                            with open(target_path, 'w') as target_file:
                                target_file.write(translated_content)
                                target_file.close()
                        else:
                            print("ERROR file", target_path)

    def translate_label(self, stories_dir, engligh_version_dir, name_to_english_dic, extract_name_to_english_dic, story_to_english_dic):
        #translator = GoogleTranslator(source='auto', target='en')
        story_total = list(name_to_english_dic.keys()) +  list(extract_name_to_english_dic.keys())
        for story_folder in story_total:
            if story_folder == ".DS_Store":
                continue
            else:
                if story_folder in name_to_english_dic:
                    name_to_english_this_story = name_to_english_dic[story_folder]
                    print("label", name_to_english_this_story)
                else:
                    name_to_english_this_story = extract_name_to_english_dic[story_folder]
                    print("extract", name_to_english_this_story)
                #translated_story_folder = translator.translate(story_folder)
                translated_story_folder = story_to_english_dic[story_folder]
                target_dir = os.path.join(engligh_version_dir, translated_story_folder)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                source_dir = os.path.join(stories_dir, story_folder)
                for background_file in os.listdir(source_dir):
                    if background_file == ".DS_Store":
                        continue
                    else:
                        name, ext = os.path.splitext(background_file)
                        if name != "all":
                            translated_file = name_to_english_this_story[name]
                        else:
                            translated_file = name
                        target_path = os.path.join(target_dir,  translated_file + ext)
                        if os.path.isfile(target_path):
                            continue
                        else:
                            print(target_path)
                            with open(os.path.join(source_dir, background_file), 'r') as f:
                                data = json.load(f)
                                f.close()
                            translated_data = {}
                            for character_name, relationships in data.items():
                                translated_name = name_to_english_this_story[character_name]
                                translated_relationships = []
                                for relationship in relationships:
                                    linked_character = name_to_english_this_story[relationship[0]]

                                    translated_relationships.append([linked_character, relationship[1]])
                                translated_data[translated_name] = translated_relationships
                                
                            with open(target_path, 'w') as f:
                                json.dump(translated_data, f, ensure_ascii=False, indent=4)
                                f.close()

    def translate_text(self, text, dest_language='en'):
        #print(text[:20])
        translator = Translator()
        try:
            translation = translator.translate(text, dest=dest_language)
            return translation.text
        except:
            try:
                sentences = text.split("\n")
                text_combine = ""
                for chunk in sentences:
                    #translation = translator.translate(chunk, dest=dest_language)
                    translation = GoogleTranslator(source='auto', target='en').translate(chunk)
                    text_combine = text_combine + "\n" + translation
                return text_combine
            except:
                return None
            
    def gen_response_35(self, sys: str, prompt: str):
        #print(prompt[:200])
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
                response = completion.choices[0].message["content"].replace('`', '').replace('json', '').replace('\n', '').replace('    ', ' ').replace('  ', ' ')  
                start_index = response.find("{")
                end_index = response.rfind("}")
                data = json.loads(response[start_index:end_index+1])
                return data

            except json.JSONDecodeError as e:
                if self.retry:
                    sys = sys + "Careful about comma in JSON format."
                    self.retry = False
                    print("JSONDecodeError", response)
                    print("JSONDecodeError, RETRY", response[start_index:end_index+1])
                    return self.gen_response_35(sys, prompt)
                else:
                    self.retry = True
                    print("JSONDecodeError: ", e)
                    print("Couldn't fix the JSON", response)
                    return {}
            
        except Exception as e:
            print("ERROR response", e)
            sleep(2)
            if self.retry:
                self.retry = False
                return self.gen_response_35(sys, prompt)
            else:
                self.retry = True
                return {}

