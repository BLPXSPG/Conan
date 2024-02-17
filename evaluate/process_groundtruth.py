import os
import json
import shutil
def find_same_as_relationships(rel_data):
    same_as = {}
    for name, links in rel_data.items():
        for linked_name, relationship in links:
            if "Same person as X (different reference)".lower() in relationship.lower():
                if name not in same_as:
                    same_as[name] = []
                same_as[name].append(linked_name)
    return same_as

def group_linked_identities(linked_data):
    grouped = []
    visited = set()

    for name, links in linked_data.items():
        if name not in visited:
            linked_group = set(links)
            linked_group.add(name)
            for link in links:
                if link in linked_data:
                    linked_group.update(linked_data[link])
                    visited.update(linked_data[link])
            grouped.append(list(linked_group))
            visited.update(linked_group)
    grouped = [sorted(i) for i in grouped]

    return grouped

def combine_same_person(relationships, grouped_identities):

    for item in grouped_identities:
        # print(item)
        same_person_relation = []
        for i in range(len(item)):
            if item[i] in relationships.keys():
                for pair in relationships[item[i]]:
                    if "Same person as X (different reference)".lower() not in pair[1].lower() and pair not in same_person_relation:
                        same_person_relation.append(pair)

        for i in range(len(item)):
            remain = [item[j] for j in range(len(item)) if j != i]
            if i == 0:
                relationships[item[i]] = same_person_relation
                for name in remain:
                    relationships[item[i]].append([name, "Same person as X (different reference)".lower()])
            else:
                relationships[item[i]] = []
                for name in remain:
                    relationships[item[i]].append([name, "Same person as X (different reference)".lower()])
    print(relationships)
    return relationships

def format_json(data, indent=4):
    def format_list(lst):
        relation = []
        for item in lst:
            relation.append(json.dumps([item[0], item[1].lower()], ensure_ascii=False))
        # relation = [json.dumps(item, ensure_ascii=False) for item in lst]
        string = ", \n      ".join(relation)
        return "[\n      " + string + "\n    ]"

    formatted_items = []
    for key, value in data.items():
        formatted_value = format_list(value)
        formatted_items.append(f'{" " * indent}"{key}": {formatted_value}')

    inner = ",\n".join(formatted_items)
    return "{\n" + inner + "\n}"

def deal_with_same_person(relationships):
    same_as = find_same_as_relationships(relationships)
    # Group the linked identities
    grouped_identities = group_linked_identities(same_as)

    print(grouped_identities)

    relationships = combine_same_person(relationships,grouped_identities)
    # formatted_cleaned_combined_relationships = format_json(relationships)
    return relationships


if __name__ == "__main__":


    root = r'..\data\chinese\label'

    for folder in os.listdir(root):
        if 'DS_Store' in folder:
            continue
        if not os.path.isdir(os.path.join(root+'_lower', folder)):
            os.makedirs(os.path.join(root+'_lower', folder))
        for file in os.listdir(os.path.join(root, folder)):
            with open(os.path.join(root, folder, file), 'r',encoding='utf-8') as f:
                relationships = json.load(f)
            full_file_name = os.path.join(root, folder, file)
            print(full_file_name)
            full_file_name = full_file_name.replace('label','label_lower')
            # formatted_cleaned_combined_relationships = deal_with_same_person(relationships)
            formatted_cleaned_combined_relationships = format_json(relationships)
            # Saving the formatted and cleaned combined relationships to a new JSON file
            with open(full_file_name, 'w', encoding='utf-8') as file:
                file.write(formatted_cleaned_combined_relationships)

    root = '../data/chinese/label_lower'

    for folder in os.listdir(root):
        if 'DS_Store' in folder:
            continue
        new_dir = root.replace('label_lower','label_v2')
        if not os.path.isdir(os.path.join(new_dir, folder)):
            os.makedirs(os.path.join(new_dir, folder))
        for file in os.listdir(os.path.join(root, folder)):
            with open(os.path.join(root, folder, file), 'r',encoding='utf-8') as f:
                relationships = json.load(f)

            full_file_name = os.path.join(root, folder, file)

            print(full_file_name)


            full_file_name = full_file_name.replace('label_lower','label_v2')

            relationships = deal_with_same_person(relationships)

            formatted_cleaned_combined_relationships = format_json(relationships)

            with open(full_file_name, 'w', encoding='utf-8') as file:
                file.write(formatted_cleaned_combined_relationships)

    shutil.rmtree(root)