import json

def load_json(file_name):
    with open(file_name,'r') as f:
        json_content=json.load(f)
    return json_content

file_path = "test_data/coco_instances_results.json"
json_content = load_json(file_path)
print('kk')