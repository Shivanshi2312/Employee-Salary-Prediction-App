import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("cleaned_data.csv")

# Encode categorical data
le_edu = LabelEncoder()
le_role = LabelEncoder()

df['Education_Level'] = le_edu.fit_transform(df['Education_Level'])
df['Job_Role'] = le_role.fit_transform(df['Job_Role'])

# Features & target
X = df[['Experience', 'Age', 'Education_Level', 'Job_Role']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()

# Save model (optional)
import pickle
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_edu, open("edu_encoder.pkl", "wb"))
pickle.dump(le_role, open("role_encoder.pkl", "wb"))