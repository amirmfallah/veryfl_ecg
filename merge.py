import pandas as pd
df1 = pd.read_csv('ecg/mitbih_test.csv', header=None)
df2 = pd.read_csv('ecg/mitbih_train.csv', header=None)

# Append the dataframes vertically
combined_df = pd.concat([df1, df2], ignore_index=True)


# Write the combined dataframe to a new CSV file
combined_df.to_csv('data1.csv', index=False)