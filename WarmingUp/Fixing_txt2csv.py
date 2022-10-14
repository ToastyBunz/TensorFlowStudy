import pandas as pd

# This will convert ; to , in text files so it is easier for CSVs
with open('E:\pythonProject\TensorReview\TF_Questions\household_power_consumption.txt') as fobj:
    with open('E:\pythonProject\TensorReview\TF_Questions\household_power_consumption.csv', 'w') as csv:
        var = fobj.read()
        var = var.replace(';', ',')
        csv.write(var)


# This drops the Data and Time columns, replaces '?' with NaN, then removes the NaNs so there is a uniform dtype
# then it creates a new CSV and re-initiates a new Pandas that has uniform dtypes so you can do calculations
df = pd.read_csv('E:/pythonProject/TensorReview/TF_Questions/household_power_consumption.csv', nrows=86400)
# infer_datetime_format=True, parse_dates={'Datetime':[0,1]}, header=0, low_memory=False,
droppers = ['Date', 'Time', 'Unnamed: 0']

for i in range(2):
    df.drop(columns=df.columns[0],
            axis=1,
            inplace=True)
    print('1')

df.replace('?', 'NaN', inplace=True)
# print(df.isna().sum())
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv('E:\pythonProject\TensorReview\TF_Questions\HPC_2.csv')