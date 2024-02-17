import os
import json
import math
import numpy as np
# Define the source directory and target directory

truth_dir = "..\\data\\chinese\\character"


def calculate_character(compute_type, is_one):
    # Initialize a list to hold paths of created files
    intersection_len_list = []
    predict_len_list = []
    truth_len_list = []
    # Traverse the directory and read JSON files
    for subdir, dirs, files in os.walk(truth_dir):
        for file in files:
            if file.endswith('.json'):

                file_path = os.path.join(subdir, file)
                if is_one:
                    if 'all.json' in file_path:
                        continue
                else:
                    if 'all.json' not in file_path:
                        continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    truth = json.load(f)
                predict_file_path = file_path.replace('character',compute_type+os.sep+'character')
                # if 'llama' in compute_type:
                #     predict_file_path = predict_file_path.replace('all.json','update_all.json')
                with open(predict_file_path, 'r', encoding='utf-8') as f:
                    predict = json.load(f)
                if 'gpt' in predict_file_path:
                    predict = predict['characters']
                intersection_len_list.append(len(set(truth) & set(predict)))
                predict_len_list.append(len(predict))
                truth_len_list.append(len(truth))


    precision = sum(intersection_len_list) / sum(predict_len_list)
    recall = sum(intersection_len_list) / sum(truth_len_list)
    f1 = 2*precision*recall / (precision + recall)

    print(' {:.3f} & {:.3f} & {:.3f}'.format(precision,recall,f1))

import os

def modify_path(old_path, new_directory):

    path, filename = os.path.split(old_path)

    new_path = os.path.join(path, new_directory, filename)

    return new_path

def calculate_character_ext(model, is_one, method, sub_method):
    # Initialize a list to hold paths of created files
    intersection_len_list = []
    predict_len_list = []
    truth_len_list = []
    # Traverse the directory and read JSON files
    for subdir, dirs, files in os.walk(truth_dir):
        for file in files:
            if file.endswith('.json'):

                file_path = os.path.join(subdir, file)
                if is_one:
                    if 'all.json' in file_path:
                        continue
                else:
                    if 'all.json' not in file_path:
                        continue
                with open(file_path, 'r', encoding='utf-8') as f:
                    truth = json.load(f)
                predict_file_path = file_path.replace('character',model+os.sep+method)
                if method != 'extract_whole_graph':
                    predict_file_path = modify_path(predict_file_path, sub_method)

                with open(predict_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                predict = []
                for key, value in data.items():
                    predict.append(key)
                    for sub_value in value:
                        predict.append(sub_value[0])
                predict = list(set(predict))
                intersection_len_list.append(len(set(truth) & set(predict)))
                predict_len_list.append(len(predict))
                truth_len_list.append(len(truth))


    precision = sum(intersection_len_list) / sum(predict_len_list)
    recall = sum(intersection_len_list) / sum(truth_len_list)
    f1 = 2*precision*recall / (precision + recall)

    print(' {:.3f} & {:.3f} & {:.3f}'.format(precision,recall,f1))



#
print('gpt4-one')
calculate_character('gpt-4', True)
calculate_character_ext('gpt-4_bef', True,'extract_whole_graph','')
calculate_character_ext('gpt-4_bef', True,'relation_extract_directly','extract')
calculate_character_ext('gpt-4_bef', True,'relation_extract_pairwisely','extract')
# calculate_character_ext('gpt-4', True,'extract_whole_graph','')
print('gpt4-all')
calculate_character('gpt-4', False)
calculate_character_ext('gpt-4_bef', False,'extract_whole_graph','')
calculate_character_ext('gpt-4_bef', False,'relation_extract_directly','extract')
calculate_character_ext('gpt-4_bef', False,'relation_extract_pairwisely','extract')

print('gpt3.5-one')
calculate_character('gpt-3.5-turbo', True)
calculate_character_ext('gpt-3.5-turbo_v4', True,'extract_whole_graph','')
calculate_character_ext('gpt-3.5-turbo_v4', True,'relation_extract_directly','extract')
calculate_character_ext('gpt-3.5-turbo_v4', True,'relation_extract_pairwisely','extract')

print('gpt3.5-all')
calculate_character('gpt-3.5-turbo', False)
calculate_character_ext('gpt-3.5-turbo_v4', False,'extract_whole_graph','')
calculate_character_ext('gpt-3.5-turbo_v4', False,'relation_extract_directly','extract')
calculate_character_ext('gpt-3.5-turbo_v4', False,'relation_extract_pairwisely','extract')

print('llama-one')
calculate_character('llama-70b', True)
calculate_character_ext('llama-70b_v3', True,'extract_whole_graph','')
calculate_character_ext('llama-70b_v3', True,'relation_extract_directly','extract')
calculate_character_ext('llama-70b_v3', True,'relation_extract_pairwisely','extract')


print('llama-all')
calculate_character('llama-70b', False)
calculate_character_ext('llama-70b_v3', False,'extract_whole_graph','')
calculate_character_ext('llama-70b_v3', False,'relation_extract_directly','extract')
calculate_character_ext('llama-70b_v3', False,'relation_extract_pairwisely','extract')
