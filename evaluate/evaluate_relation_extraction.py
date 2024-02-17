import json
from process_groundtruth import find_same_as_relationships, group_linked_identities



file_path  = r'../data/equivalent_relation.json'
with open(file_path, 'r', encoding='utf-8') as file:
    equivalent_relation_data_ = json.load(file)
equivalent_relation_data = equivalent_relation_data_['equivalent relations']
category_relation_data = equivalent_relation_data_['High-Level relations']

for key, value in category_relation_data.items():
    for sub_value in value:
        if sub_value in equivalent_relation_data.keys():
            category_relation_data[key] += equivalent_relation_data[sub_value]

def merge_dicts(dicts):
    merged_dict = {}
    for key, value in dicts.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value
    return merged_dict

def replace_multiple_relations_to_lowercase(character_data, equivalent_relation_data):
    new_character_data = {}

    for character, relationships in character_data.items():
        new_relationships = []
        for relationship in relationships:
            new_relation = [relationship[0]]  # keep the first element (the related character) unchanged
            # Handle multiple relations in a string, separated by commas
            for relation in relationship[1:]:
                new_relations = []
                for sub_relation in relation.split(', '):  # Splitting the string by ', '
                    lower_sub_relation = sub_relation.lower()
                    replaced = False
                    for key, equivalents in equivalent_relation_data.items():
                        lower_key = key.lower()
                        lower_equivalents = [equiv.lower() for equiv in equivalents]
                        if lower_sub_relation in lower_equivalents:
                            # Replace the equivalent sub-relation with the key relation, all in lowercase
                            new_relations.append(lower_key)
                            replaced = True
                            break
                    if not replaced:
                        # If the sub-relation is not in the equivalents, just convert it to lowercase
                        new_relations.append(lower_sub_relation)
                new_relation.append(', '.join(new_relations))  # Rejoin the sub-relations
            new_relationships.append(new_relation)
        new_character_data[character] = new_relationships
    return new_character_data

def convert_to_triplets(character_data):
    triplets = []

    for character, relationships in character_data.items():
        for relationship in relationships:
            related_character = relationship[0]
            # Split the relations by comma and create triplets
            for relation in relationship[1:]:
                for sub_relation in relation.split(', '):
                    sub_relation = sub_relation.lower().replace('-',' ')
                    character = character.replace('•','·')
                    related_character = related_character.replace('•','·')
                    triplets.append((character, related_character, sub_relation.strip()))
    return triplets

def extract_unique_dyads(character_data):
    dyads = set()

    for character, relationships in character_data.items():
        for relationship in relationships:
            related_character = relationship[0]
            # Add the dyad to the set for uniqueness
            dyads.add((character, related_character))

    return list(dyads)

def extract_unique_characters(character_data):
    characters = set()

    for character, relationships in character_data.items():
        # Add the main character
        characters.add(character)
        for relationship in relationships:
            # Add the related character
            characters.add(relationship[0])

    return list(characters)

import os

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def get_predict2groundtruth(predict_file_list,ground_truth_file_list):

    predict2groundtruth = {}
    predict2detail = {}
    def find_match_groundtruth(script_name, json_name,ground_truth_file_list):
        for file in ground_truth_file_list:
            if script_name in file and json_name in file:
                return file

    for file in predict_file_list:
        file_split = file.split('\\')
        json_name = file_split[-1]
        if '-' not in file_split[-2]:
            sub_method = file_split[-2]
            script_name = file_split[-3]
            method_type = file_split[-4]
            model_type = file_split[-5]
        else:
            sub_method = ''
            script_name = file_split[-2]
            method_type = file_split[-3]
            model_type = file_split[-4]

        predict2groundtruth[file] = find_match_groundtruth(script_name, json_name,ground_truth_file_list)
        predict2detail[file] = {'json_name': json_name, 'sub_method': sub_method, 'script_name': script_name, 'method_type' : method_type, 'model_type': model_type}
    return predict2groundtruth, predict2detail


def duplicate_the_same(grouped_identities, relation):

    def find_remain_group(name):
        item = []
        for group in grouped_identities:
            if name in group:
                item = group
                break

        return item if item != [] else [name]

    g_i_m = [item for sublist in grouped_identities for item in sublist]
    new_relation = []
    for triple in relation:
        new_relation.append(triple)
        if triple[0] in g_i_m or triple[1] in g_i_m:
            c1_same = find_remain_group(triple[0])
            c2_same = find_remain_group(triple[1])
            for c1 in c1_same:
                for c2 in c2_same:
                    if (c1,c2, triple[2]) not in new_relation:
                        new_relation.append((c1,c2, triple[2]))

    return sorted(new_relation)

result_one = []
result_all = []

def calculate_f1(predict_truth_path,ground_truth_path):
    result = []
    result_for_triple = []

    with open(predict_truth_path, 'r', encoding='utf-8') as file:
        predict_truth = json.load(file)
    with open(ground_truth_path, 'r', encoding='utf-8') as file:
        ground_truth = json.load(file)

    characters = [predict_truth, ground_truth]

    use_equivalent_relationship = False

    characters_triple_v1 = []
    for item in characters:
        characters_triple_v1.append(convert_to_triplets(item))


    # get relation triple
    characters_triple_v2 = []
    for item in characters:
        characters_triple_v2.append(
            convert_to_triplets(replace_multiple_relations_to_lowercase(item, equivalent_relation_data)))

    # get relation triple_category
    characters_triple_v3 = []
    for item in characters:
        characters_triple_v3.append(
            convert_to_triplets(replace_multiple_relations_to_lowercase(item, category_relation_data)))


    predict_num = len(characters_triple_v1[0])
    ground_truth_num = len(characters_triple_v1[1])

    same_as = find_same_as_relationships(ground_truth)
    grouped_identities = group_linked_identities(same_as)

    characters_triple_v1[1] = duplicate_the_same(grouped_identities, characters_triple_v1[1])
    characters_triple_v2[1] = duplicate_the_same(grouped_identities, characters_triple_v2[1])
    characters_triple_v3[1] = duplicate_the_same(grouped_identities, characters_triple_v3[1])

    # get relation pair
    characters_pair = []
    for item in characters:
        characters_pair.append(
            extract_unique_dyads(replace_multiple_relations_to_lowercase(item, equivalent_relation_data)))

    # get characters
    characters_one = []
    for item in characters:
        characters_one.append(
            sorted(extract_unique_characters(replace_multiple_relations_to_lowercase(item, equivalent_relation_data))))

    # calculate triple level agreement
    intersection_set = set(characters_triple_v1[0]) & set(characters_triple_v1[1])
    if len(characters_triple_v1[0]) == 0:
        precision = 0
    else:
        precision = len(intersection_set) / len(characters_triple_v1[0])
    recall = len(intersection_set) / ground_truth_num
    if precision + recall == 0:
        f1=0
    else:
        f1 = 2*precision*recall / (precision + recall)
    result_for_triple.append(len(intersection_set))

    result.append([precision, recall, f1])

    # calculate triple level agreement
    intersection_set = set(characters_triple_v2[0]) & set(characters_triple_v2[1])
    if len(characters_triple_v2[0]) == 0:
        precision = 0
    else:
        precision = len(intersection_set) / len(characters_triple_v2[0])

    recall = len(intersection_set) / ground_truth_num
    if precision + recall == 0:
        f1=0
    else:
        f1 = 2*precision*recall / (precision + recall)
    result.append([precision, recall, f1])
    result_for_triple.append(len(intersection_set))


    # calculate triple level agreement
    intersection_set = set(characters_triple_v3[0]) & set(characters_triple_v3[1])
    if len(characters_triple_v3[0]) == 0:
        precision = 0
    else:
        precision = len(intersection_set) / len(characters_triple_v3[0])

    recall = len(intersection_set) / ground_truth_num
    if precision + recall == 0:
        f1=0
    else:
        f1 = 2*precision*recall / (precision + recall)
    result_for_triple.append(len(intersection_set))
    result.append([precision, recall, f1])

    result_for_triple.append(len(characters_triple_v3[0]))
    result_for_triple.append(ground_truth_num)

    return result, result_for_triple

def calculate_triple_f1(result):
    intersection = [0,0,0]
    predict_num = [0,0,0]
    ground_truth_num = [0,0,0]
    for item in result:
        for i in range(len(item)):
            intersection[i] += item[i][0]
            predict_num[i] += item[i][1]
            ground_truth_num[i] += item[i][2]


from collections import defaultdict
import pandas as pd

if __name__ == '__main__':

    # for dir_name in ['gpt-4_bef','gpt-3.5-turbo_bef','llama-70b_bef', 'llama-70b_v3','gpt-4_v4','gpt-3.5-turbo_v4',]:
    # for dir_name in ['gpt-4_bef', 'llama-70b_v3','gpt-3.5-turbo_v4',]:
    for dir_name in ['gpt-4_bef', 'gpt-3.5-turbo_v4','llama-70b_v3',]:
    # for dir_name in ['gpt-4_bef']:
        ground_truth_dir = '..\\data\\chinese\\label_v2'
        # ground_truth_dir = 'E:\Pycharm\MurderMysteryGame\data\english\label_v2'
        ground_truth_file_list = find_json_files(ground_truth_dir)
        # predict_dir = 'E:\Pycharm\MurderMysteryGame\data\english\\' + dir_name
        predict_dir = '..\\data\\chinese\\' + dir_name
        predict_file_list = find_json_files(predict_dir)
        predict2groundtruth, predict2detail = get_predict2groundtruth(predict_file_list,ground_truth_file_list)
        groundtruth2result = defaultdict(list)
        for predict_file,groundtruth_file in predict2groundtruth.items():
            print(predict_file)

            result,result_for_triple = calculate_f1(predict_file, groundtruth_file)
            name = predict2detail[predict_file]['script_name']+'_'+predict2detail[predict_file]['json_name']
            method = predict2detail[predict_file]['method_type']+'_'+predict2detail[predict_file]['sub_method'] if predict2detail[predict_file]['sub_method'] != '' else predict2detail[predict_file]['method_type']
            groundtruth2result[name].append([method, result,result_for_triple])
            # break

        full2abbr = {
            'extract_whole_graph': 'w',
            'relation_extract_directly_extract': 'd_e',
            'relation_extract_directly_given': 'd_g',
            'relation_extract_pairwisely_extract': 'p_e',
            'relation_extract_pairwisely_given': 'p_g',

        }
        print(groundtruth2result)
        result_list_all = []
        result_list_one = []
        for name, scores in groundtruth2result.items():
            result_dict = {}
            result_dict['name'] = name
            for item in scores:
                method,result,result_for_triple = item
                method = full2abbr[method]
                result_dict[method+'_v1_p'] = result[0][0]
                result_dict[method+'_v1_r'] = result[0][1]
                result_dict[method+'_v1_f1'] = result[0][2]
                result_dict[method+'_v2_p'] = result[1][0]
                result_dict[method+'_v2_r'] = result[1][1]
                result_dict[method+'_v2_f1'] = result[1][2]
                result_dict[method+'_v3_p'] = result[2][0]
                result_dict[method+'_v3_r'] = result[2][1]
                result_dict[method+'_v3_f1'] = result[2][2]
                result_dict[method+'_v1_tn'] = result_for_triple[0]
                result_dict[method+'_v2_tn'] = result_for_triple[1]
                result_dict[method+'_v3_tn'] = result_for_triple[2]
                result_dict[method+'_pn'] = result_for_triple[3]
                result_dict[method+'_gn'] = result_for_triple[4]

            if 'all' in name:
                result_list_all.append(result_dict)
            else:
                result_list_one.append(result_dict)

        data = pd.DataFrame(result_list_all)
        mean_values = data.drop('name', axis=1).mean()
        new_row = pd.Series(data={'name': 'Average'}, name='Average').append(mean_values)
        data = data.append(new_row, ignore_index=True)

        all_values = data[data['name'] != 'Average'].drop('name', axis=1).sum()
        all_values_row = pd.Series(data={'name': 'sum'}, name='sum').append(all_values)


        new_row_triple = pd.Series(data={'name': 'triple'}, name='triple').append(mean_values)
        for method in full2abbr.values():
            for type in ['_v1', '_v2', '_v3']:
                precision = all_values[method+type+'_tn'] / all_values[method+'_pn']
                recall = all_values[method+type+'_tn'] / all_values[method+'_gn']
                f1 = 2*precision*recall / (precision + recall)
                new_row_triple[method+type+'_p'] = precision
                new_row_triple[method+type+'_r'] = recall
                new_row_triple[method+type+'_f1'] = f1
        data = data.append(new_row_triple, ignore_index=True)
        # data = data.append(all_values_row, ignore_index=True)
        data = data[[item for item in data.columns if ('n' not in item) or (item == 'name')]]
        # data = data[['name',  'w_v2_f1',  'd_e_v2_f1', 'p_e_v2_f1' ]]
        # data = data[['name','w_v2_p', 'w_v2_r', 'w_v2_f1', 'd_e_v2_p', 'd_e_v2_r', 'd_e_v2_f1',  'p_e_v2_p', 'p_e_v2_r', 'p_e_v2_f1']]
        data = data[['name','w_v1_p', 'w_v1_r', 'w_v1_f1', 'w_v2_p', 'w_v2_r', 'w_v2_f1', 'w_v3_p', 'w_v3_r', 'w_v3_f1', 'd_e_v1_p', 'd_e_v1_r', 'd_e_v1_f1', 'd_e_v2_p', 'd_e_v2_r', 'd_e_v2_f1', 'd_e_v3_p', 'd_e_v3_r', 'd_e_v3_f1',  'p_e_v1_p', 'p_e_v1_r', 'p_e_v1_f1', 'p_e_v2_p', 'p_e_v2_r', 'p_e_v2_f1', 'p_e_v3_p', 'p_e_v3_r', 'p_e_v3_f1']]
        # data = data[['name', 'd_g_v2_p', 'd_g_v2_r', 'd_g_v2_f1',  'p_g_v2_p', 'p_g_v2_r', 'p_g_v2_f1']]

        csv_file_path = r'.\\result\\'+dir_name+'_all.xlsx'
        data.round(3).to_excel(csv_file_path, index=False,  encoding='utf-8-sig')

        data = pd.DataFrame(result_list_one)
        mean_values = data.drop('name', axis=1).mean()
        new_row = pd.Series(data={'name': 'Average'}, name='Average').append(mean_values)
        data = data.append(new_row, ignore_index=True)

        all_values = data[data['name'] != 'Average'].drop('name', axis=1).sum()
        all_values_row = pd.Series(data={'name': 'sum'}, name='sum').append(all_values)

        new_row_triple = pd.Series(data={'name': 'triple'}, name='triple').append(mean_values)
        for method in full2abbr.values():
            for type in ['_v1', '_v2', '_v3']:
                precision = all_values[method+type+'_tn'] / all_values[method+'_pn']
                recall = all_values[method+type+'_tn'] / all_values[method+'_gn']
                f1 = 2*precision*recall / (precision + recall)
                new_row_triple[method+type+'_p'] = precision
                new_row_triple[method+type+'_r'] = recall
                new_row_triple[method+type+'_f1'] = f1
        data = data.append(new_row_triple, ignore_index=True)
        print(data.columns)
        # data = data.append(all_values_row, ignore_index=True)
        data = data[[item for item in data.columns if ('n' not in item) or (item == 'name')]]
        # data = data[['name',  'w_v2_f1',  'd_e_v2_f1',  'p_e_v2_f1' ]]
        # data = data[['name','w_v2_p', 'w_v2_r', 'w_v2_f1', 'd_e_v2_p', 'd_e_v2_r', 'd_e_v2_f1',  'p_e_v2_p', 'p_e_v2_r', 'p_e_v2_f1']]
        # data = data[['name', 'd_g_v2_p', 'd_g_v2_r', 'd_g_v2_f1',  'p_g_v2_p', 'p_g_v2_r', 'p_g_v2_f1']]
        data = data[['name','w_v1_p', 'w_v1_r', 'w_v1_f1', 'w_v2_p', 'w_v2_r', 'w_v2_f1', 'w_v3_p', 'w_v3_r', 'w_v3_f1', 'd_e_v1_p', 'd_e_v1_r', 'd_e_v1_f1', 'd_e_v2_p', 'd_e_v2_r', 'd_e_v2_f1', 'd_e_v3_p', 'd_e_v3_r', 'd_e_v3_f1',  'p_e_v1_p', 'p_e_v1_r', 'p_e_v1_f1', 'p_e_v2_p', 'p_e_v2_r', 'p_e_v2_f1', 'p_e_v3_p', 'p_e_v3_r', 'p_e_v3_f1']]

        csv_file_path = r'.\\result\\'+dir_name +'_one.xlsx'
        data.round(3).to_excel(csv_file_path, index=False,  encoding='utf-8-sig')

