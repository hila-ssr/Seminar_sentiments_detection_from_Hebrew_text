import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the CSV file into a DataFrame
file_path = 'ANSWERS_comereation.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Assuming your CSV file has columns 'model_list' and 'y_list'
predicted_statuses = df['GPT_ans'].tolist()
actual_statuses = df['y_list'].tolist()

# Create confusion matrix
conf_matrix = confusion_matrix(actual_statuses, predicted_statuses, labels=['Positive', 'Neutral', 'Negative'])

# Create a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Neutral', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Neutral', 'Predicted Negative'])

# Print the confusion matrix with titles
print("Confusion Matrix:")
print(conf_matrix_df)
print('\n\n')
# Print classification report for additional metrics
print(df['GPT'].value_counts() )
print('\n\n')
print("\nClassification Report:")
print(classification_report(actual_statuses, predicted_statuses, labels=['Positive', 'Neutral', 'Negative']))


import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the CSV file into a DataFrame
file_path = 'ANSWERS_comereation.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Assuming your CSV file has columns 'model_list' and 'y_list'
predicted_statuses = df['Hobert_ans'].tolist()
actual_statuses = df['y_list'].tolist()

# Create confusion matrix
conf_matrix = confusion_matrix(actual_statuses, predicted_statuses, labels=['Positive', 'Neutral', 'Negative'])

# Create a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Positive', 'Actual Neutral', 'Actual Negative'], columns=['Predicted Positive', 'Predicted Neutral', 'Predicted Negative'])

# Print the confusion matrix with titles
print("Confusion Matrix:")
print(conf_matrix_df)
print('\n\n')
# Print classification report for additional metrics
print(df['HeBERT'].value_counts() )
print('\n\n')
print("\nClassification Report:")
print(classification_report(actual_statuses, predicted_statuses, labels=['Positive', 'Neutral', 'Negative']))
