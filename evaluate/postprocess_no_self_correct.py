import json
import re

file_path  = r'../data/equivalent_relation.json'
with open(file_path, 'r', encoding='utf-8') as file:
    equivalent_relation_data_ = json.load(file)
equivalent_relation_data = equivalent_relation_data_['equivalent relations']

combined_list = []


# Adding keys and values to the list
for key, values in equivalent_relation_data.items():
    combined_list.append(key)
    combined_list.extend(values)
combined_list = [i.lower().replace('-',' ') for i in combined_list]
print(combined_list)


before_list = [[],[],[]]
after_list = [[],[],[]]
def format_json(data, indent=4):
    def format_list(lst):
        return "[\n      " + ", \n      ".join(json.dumps(item, ensure_ascii=False) for item in lst) + "\n    ]"

    formatted_items = []
    for key, value in data.items():
        formatted_value = format_list(value)
        formatted_items.append(f'{" " * indent}"{key}": {formatted_value}')

    inner = ",\n".join(formatted_items)
    return "{\n" + inner + "\n}"

all_relation = []


def repair_json(extract):
    if "{" in extract:
        index_position = [i for i, c in enumerate(extract) if c == "{"]
        index_position.reverse()
    else:
        return None
    for pos in index_position:
        data = extract[pos:]
        for i in range(len(data), 0, -1):
            try:
                # Try to parse the JSON
                character_data = data[:i]
                if '}' not in data:
                    character_data = character_data + '}'
                parsed = json.loads(character_data)
                # If successful, return the parsed JSON
                return parsed
            except json.JSONDecodeError:
                # If there's a JSON decode error, try with a shorter string
                continue

    return None

def postprocessing(character_data, replace_chinese,mathod_type):
    if isinstance(character_data, str):
        character_data = character_data.strip().replace('\n','')
        try:
            character_data = eval(character_data)
        except:
            character_data = repair_json(character_data)
            if character_data is None:
                return
    new_dict = {}
    for character, relationships in character_data.items():
        new_dict[character] = []
        for relationship in relationships:
            tmp_relations = []
            if isinstance(relationship, dict):
                continue
            if all(isinstance(item, list) for item in relationship):
                try:
                    format_relationship = [relationship[0][0]]
                    for sub_list in relationship:
                        format_relationship += sub_list[1:]
                    relationship = format_relationship
                except:
                    continue
            if all(isinstance(item, str) for item in relationship):
                if len(relationship) == 1:
                    continue
                related_character = relationship[0]
                character_relation=[]

                for relation in relationship[1:]:
                    for sub_relation in re.split(r',|、|，|\.', relation):
                        sub_relation = sub_relation.strip()
                        sub_relation = sub_relation.lower()
                        sub_relation = sub_relation.replace('-',' ')
                        if 'stranger' in sub_relation:
                            continue
                        if sub_relation == " " or sub_relation == '':
                            continue
                        if mathod_type == 0:
                            before_list[0].append(sub_relation)
                        if mathod_type == 1:
                            before_list[1].append(sub_relation)
                        if mathod_type == 2:
                            before_list[2].append(sub_relation)

                        sub_relation = sub_relation.replace(related_character,'x').replace(character,'x')
                        if ' of' in  sub_relation and ' of x' not in sub_relation and sub_relation not in ["Perpetrator of X's family".lower(), "Seeker of Help from X".lower()] :
                            sub_relation = sub_relation.replace(' of', ' of x')
                        if ' to' in  sub_relation and ' to x' not in sub_relation:
                            sub_relation = sub_relation.replace(' of', ' of x')
                        if ' by' in  sub_relation and ' by x' not in sub_relation:
                            sub_relation = sub_relation.replace(' of', ' of x')
                        if 'x' not in sub_relation:
                            sub_relation = sub_relation + ' of x'

                        if sub_relation in combined_list:
                            if mathod_type == 0:
                                after_list[0].append(sub_relation)
                            if mathod_type == 1:
                                after_list[1].append(sub_relation)
                            if mathod_type == 2:
                                after_list[2].append(sub_relation)
                            tmp_relations.append(sub_relation)
                        else:
                            all_relation.append(sub_relation)

            if len(tmp_relations) != 0:
                character_relation.append(related_character)
                character_relation.append(', '.join(tmp_relations))
                new_dict[character].append(character_relation)
    filtered_data = {k: v for k, v in new_dict.items() if v}

    filtered_data = format_json(filtered_data)
    return filtered_data


import os
from pathlib import Path
language = 'english'

folder_path = r'../data/'+language+'/llama-70b'

if folder_path.split('\\')[-1] == 'gpt-3.5-turbo':
    origin = 'gpt-3.5-turbo'
    replace = 'gpt-3.5-turbo_bef'
elif folder_path.split('\\')[-1] == 'llama-70b':
    origin = 'llama-70b'
    replace = 'llama-70b_bef'
elif folder_path.split('\\')[-1] == 'gpt-4':
    origin = 'gpt-4'
    replace = 'gpt-4_bef'

label_list = os.listdir('../data/'+language+'/label')
folder_path_detail = [os.path.join(folder_path, i) for i in os.listdir(folder_path) if i != 'character']

for folder in folder_path_detail:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                full_path = str(Path(root) / file)
                flag = False
                for item in label_list:
                    if item in full_path:
                        flag = True
                if not flag:
                    continue

                print(full_path)
                with open(full_path, 'r', encoding='utf-8') as file:
                    predict_truth = json.load(file)
                if 'extract_whole_graph' in full_path:
                    mathod_type = 0
                elif 'relation_extract_directly' in full_path:
                    mathod_type = 1
                else:
                    mathod_type = 2
                new_dict = postprocessing(predict_truth,False, mathod_type)
                desired_path = full_path.rsplit('\\', 1)[0] + '\\'
                desired_path = desired_path.replace(origin,replace)

                if not os.path.isdir(desired_path):
                    os.makedirs(os.path.join(desired_path))

                full_file_name = full_path.replace(origin,replace).replace('update_all','all')

                with open(full_file_name, 'w', encoding='utf-8') as file:
                    if new_dict is None:
                        new_dict = "{}"
                    file.write(new_dict)

print(len(all_relation))
print(1-len(after_list[0])/len(before_list[0]))
print(1-len(after_list[1])/len(before_list[1]))
print(1-len(after_list[2])/len(before_list[2]))
