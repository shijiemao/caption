from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
file = pd.read_csv("result.csv")
string = file["String"]
label = file["Label"]
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

task_prefix = "summarize: "
# use different length sentences to test batching


inputs = tokenizer([task_prefix + sentence for sentence in label], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)
all = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
print(all)
# import csv
# import csv
# from itertools import zip_longest

# question = file['String']
# d = [question, all]
# export_data = zip_longest(*d, fillvalue = '')
# with open('result_4.csv', 'w', encoding="utf-8", newline='') as myfile:
#       wr = csv.writer(myfile)
#       wr.writerow(("String", "new_string"))
#       wr.writerows(export_data)
# myfile.close()
