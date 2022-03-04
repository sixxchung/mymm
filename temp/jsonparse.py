### JSON (JavaScript Object Notation)
# https://docs.python.org/3/library/json.html#json.dumps

import json

# Read string
json_string='{"title": "example", "age": 38, "married": true}'
contents = json.loads(json_string)
# pretty
print(json.dumps(contents, sort_keys=True ,indent=2))

# Load file
with open('./text.json') as json_file:
    contents = json.load(json_file)

print(json.dumps(contents, indent=4, ensure_ascii=False))

# Write JSON 
with open('./data.json', 'w') as json_file:
    json.dump(json_string, json_file)

# parse 1 value
val = contents["title"]
val = contents.get("title")
print(f'value=> {val}')

# json  to  python
# object to dictionaries
# arrays to lists
# Null to None
valList = contents.get("house").get("json_array")
for val in valList:
    print(val)

valDic = contents.get("house").get("json_object")
valDic.keys()
valDic.values()
for key in valDic:
    print(key, valDic[key])
for key, val in valDic.items():
    print(key, val)
for item in valDic.items():
    print(item)

import jmespath
jmespath.search("persons[*].age", contents)
jmespath.search("house.json_object.name", contents)
