import json
from os import path
import collections as cl

def output_json(json_filename, comment_list, true_list, pred_list):
    output = []
    for i in range(len(comment_list)):
        data = cl.OrderedDict()
        data["comment"] = comment_list[i]
        data["treu"] = true_list[i]
        data["pred"] = pred_list[i]
        output.append(data)

    with open(json_filename, mode='w') as f:
        json.dump(output, f, sort_keys=True, indent=4)

test_texts = ["a", "b", "c"]
y_test = [0,1,2]
y_pred = [1,1,2]

output_json_filename = path.join(path.dirname(__file__), "output.json")
output_json(output_json_filename, test_texts, y_test, y_pred)
