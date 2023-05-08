

import os
import openai
import wandb

import os
  
# Get the value of 'HOME'
# environment variable
key = 'OPENAI_API_KEY'
value = os.getenv(key)
  
# Print the value of 'HOME'
# environment variable
print("Value of 'HOME' environment variable :", value) 
print(os.environ)
# OPENAI_API_KEY='HOME'
# openai.api_key = os.getenv(OPENAI_API_KEY)
# print(openai.api_key)
# run = wandb.init(project='GPT-3 in Python')
# prediction_table = wandb.Table(columns=["prompt", "completion"])
# gpt_prompt = "Correct this to standard English:\n\nShe no went to the market."


# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=gpt_prompt,
#   temperature=0.5,
#   max_tokens=256,
#   top_p=1.0,
#   frequency_penalty=0.0,
#   presence_penalty=0.0
# )


# print(response['choices'][0]['text'])


# prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
