# imports
import numpy as np
import pandas as pd
import codecs
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# function to load data
def load_data(filename):
    i = 0
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    # print(x[:10] , ' ',y[:10])
    x = list(x)
    y = list(y)  # to_categorical(y, 3)

    return x, y


# load data
x_token_train, y_token_train = load_data('token_train.txt')

filtered_x_token_train_values=[]
filtered_y_token_train_values=[]
filtered_indices = []

#הגודל של הפילטור
for i in range(len(x_token_train)):
    if len(x_token_train[i]) < 300:
        filtered_x_token_train_values.append(x_token_train[i])
        filtered_y_token_train_values.append(y_token_train[i])
        filtered_indices.append(i)

openai_api_key = os.environ.get('SECRET_KEY')
chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
y_model_list = []
y_model_index = []
index = 0

for i in filtered_x_token_train_values:
    y_model_index.append(index)

    messages = [
        SystemMessage(
            content="you are a NLP system that help detect if the emotion in the text is Positive, Negative or Neutral."
        ),
        HumanMessage(
            content="what is the emotion in the next sentence : {}? please write as answer only Positive, Negative or Neutral".format(
                i)
        ),
    ]
    response = chat(messages)
    y_model_list.append(response.content)
    print(index, ' ', response)
    index = index + 1

# Create a DataFrame
df = pd.DataFrame({'sentiments': y_model_list[:8089], 'y_model_index': y_model_index[:8089]})

# Specify the CSV file name
csv_file_name = "output_GPT_answers.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_name, index=False)

print(f"Data has been written to {csv_file_name}")

#נחליף את הנתונים של העמודת Y המקורית
filtered_y_token_train_values_words = ['Positive' if value == '0' else 'Negative' if value == '1' else 'Neutral' for value in filtered_y_token_train_values]

# Creating a DataFrame with columns from both lists
df = pd.DataFrame({
    'sentiments': y_model_list[:8089],
    'filtered_y_token_train_values_words': filtered_y_token_train_values_words[:8089]
})

# Adding a check column to verify equality between the two columns
df['check'] = df['sentiments'] == df['filtered_y_token_train_values_words']



#נבדוק כמה נתונים נכונים או לא נכונים היו
check_counts = df['check'].value_counts()

print("Counts of True and False:")
print(check_counts)

# Saving the result
df = pd.DataFrame({'sentiments': y_model_list[:8089], 'y_token_train': y_model_index[:8089]})

# Specify the CSV file name
csv_file_name = "output_GPT_comereation.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_name, index=False)

print(f"Data has been written to {csv_file_name}")