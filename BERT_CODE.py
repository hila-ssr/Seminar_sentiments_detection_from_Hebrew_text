## to  run the next code we need to run it on Google Colab Drive

#imports
import numpy as np
import pandas as pd
import codecs
from google.colab import drive
drive.mount('/content/drive')
#every time we run the code we need to run the next row to install the transformers library
#!pip install transformers



#function to load data
def load_data(filename):
    i = 0
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    x =list(x)
    y  = list(y)

    return x, y

#load data
x_token_train, y_token_train = load_data('token_train.txt')



#load model bert
from transformers import AutoTokenizer, AutoModel, pipeline
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")

# how to use?
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="avichr/heBERT_sentiment_analysis",
    tokenizer="avichr/heBERT_sentiment_analysis",
    return_all_scores = True
)


#filter all the long sentenses (longer than 300) because it was too hrd to the models to calculate
filtered_x_token_train_values = []  # List to store values that satisfy the condition for x_token_train
filtered_y_token_train_values = []  # List to store corresponding values from y_token_train
filtered_indices = []  # List to store corresponding indices

#הגודל של הפילטור
for i in range(len(x_token_train)):
    if len(x_token_train[i]) < 300:
        filtered_x_token_train_values.append(x_token_train[i])
        filtered_y_token_train_values.append(y_token_train[i])
        filtered_indices.append(i)


        #we will create chunks to see if or when the code is falling
chunk_size = 100  # Define the size of each chunk

# Split x_token_train into smaller chunks
chunks = [filtered_x_token_train_values[i:i + chunk_size] for i in range(0, len(filtered_x_token_train_values), chunk_size)]

santiment = []
for chunk in chunks:
    chunk_sentiments = []  # Store sentiments for each chunk
    for item in chunk:
        max_label = max(sentiment_analysis(item)[0], key=lambda x: x['score'])
        chunk_sentiments.append(max_label['label'])
        print(item)
    santiment.extend(chunk_sentiments)  # Add chunk sentiments to the main sentiment list


    #replace the original values of the y column
filtered_y_token_train_values1 =   ['positive' if x == '0' else 'negative' if x == '1' else 'netural' for x in filtered_y_token_train_values]

# Creating a DataFrame with columns from both lists
df = pd.DataFrame({
    'sentiments': santiment,
    'filtered_y_token_train_values': filtered_y_token_train_values1
})

# Adding a check column to verify equality between the two columns
df['check'] = df['sentiments'] == df['filtered_y_token_train_values']


#we will check how many answers were corrent and how many were not
check_counts = df['check'][:8089].value_counts()

print("Counts of True and False:")
print(check_counts)


# Specify the CSV file name
csv_file_name = "HeBert_answers_with_comperation1.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file_name, index=False)

print(f"Data has been written to {csv_file_name}")