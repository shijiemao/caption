import requests
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/voidful/bart-eqg-question-generator"
API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

file = 	pd.read_csv("result.csv")
string = file["Label"]
all = []
for i in string:
	
     output = query({
	    "inputs":i,
     })
     all.append(output)
     print(output)



import csv
import csv
from itertools import zip_longest

question = file['String']
d = [question, all]
export_data = zip_longest(*d, fillvalue = '')
with open('result_2.csv', 'w', encoding="utf-8", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("String", "new_string"))
      wr.writerows(export_data)
myfile.close()
