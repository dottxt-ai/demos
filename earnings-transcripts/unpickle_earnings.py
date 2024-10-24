import pickle
import os

with open('motley-fool-data.pkl', 'rb') as file:
    # Load the object from the file
    loaded_object = pickle.load(file)

print(loaded_object.head())

# Make names
loaded_object['name'] = loaded_object['ticker'] + '-' + loaded_object['q']

# Create the transcripts directory
os.makedirs('transcripts', exist_ok=True)

# Go through each row and save it to transcripts/{name}.txt
for index, row in loaded_object.iterrows():
    with open(f'transcripts/{row["name"]}.txt', 'w') as file:
        file.write("Company name: " + row['name'] + "\n")
        file.write("Company ticker: " + row['ticker'] + "\n")
        file.write("Earnings call date: " + str(row['date']) + "\n") # date is sometimes a list
        file.write("Earnings call quarter: " + row['q'] + "\n")
        file.write("Earnings call transcript: " + "\n")
        file.write(row['transcript'].replace('\n', '\n\n'))
