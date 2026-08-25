import pandas as pd

cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
        "BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv("diabetes.csv", header=None, names=cols)
df.to_csv("diabetes_raw.csv", index=False)
print(df.shape)
print(df.head())
print(df.dtypes)
