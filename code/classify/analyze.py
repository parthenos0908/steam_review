import json
import csv
from os import path


id = 227300
# id = 255710

def main():
    r_filename = path.join(path.dirname(__file__), "data/" + str(id) + "/" + str(id) + "_review_predict.json")
    with open(r_filename, mode='r') as f:
        review_pred = json.load(f)
    f_filename = path.join(path.dirname(__file__), "data/" + str(id) + "/" + str(id) + "_forum_predict.json")
    with open(f_filename, mode='r') as f:
        forum_pred = json.load(f)
    c_filename = path.join(path.dirname(__file__), "data/" + str(id) + "/" + str(id) + "_cross_predict.json")
    with open(c_filename, mode='r') as f:
        cross_pred = json.load(f)

    answer_list = []
    r_pred_list = []
    f_pred_list = []
    cross_pred_list = []
    review_list = []
    for i in range(len(review_pred)):
        answer_list.append(review_pred[i]["answer"])
        r_pred_list.append(review_pred[i]["pred"])
        f_pred_list.append(forum_pred[i]["pred"])
        cross_pred_list.append(cross_pred[i]["pred"])
        review_list.append(review_pred[i]["review"])


    csv_filename = path.join(path.dirname(__file__), str(id) + "_result.csv")
    with open(csv_filename, "w", newline="", encoding='CP932', errors='replace') as f:
        writer = csv.writer(f)
        writer.writerow(["answer", "r_pred", "f_pred", "cross_pred", "review"])
        for i in range(len(review_pred)):
            writer.writerow([answer_list[i], r_pred_list[i], f_pred_list[i], cross_pred_list[i], review_list[i]])

if __name__ == '__main__':
    main()