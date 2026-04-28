import pandas as pd

# Load data
df = pd.read_csv("employee_data.csv")

print("Original Data:\n", df.head())

# Handle missing values
df.fillna({
    'Experience': df['Experience'].mean(),
    'Age': df['Age'].mean(),
    'Education_Level': 'Bachelor',
    'Job_Role': 'Junior',
    'Salary': df['Salary'].mean()
}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned data saved as cleaned_data.csv")