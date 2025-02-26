import json

import pandas as pd

ds = pd.read_csv("data.csv", encoding="utf-8")

lines = ds.to_dict(orient="records")

wfile = open("data.txt", "w")
for line in lines:
    new_line = line["title"] + "，" + line["ask"] + "，" + line["answer"]
    new_line = {"text": new_line}
    wfile.write(json.dumps(new_line, ensure_ascii=False) + "\n")
