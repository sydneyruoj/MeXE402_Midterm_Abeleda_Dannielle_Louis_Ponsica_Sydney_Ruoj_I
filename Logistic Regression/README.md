# Logistic Regression of Pima Indians Diabetes
The **Pima Indians Diabetes Database** project aims to analyze factors contributing to diabetes onset in women of Pima Indian heritage. The dataset includes medical data such as glucose levels, BMI, insulin, age, and pregnancy history. The objective is to use this data to build predictive models, such as logistic regression, to classify patients as diabetic or non-diabetic based on these attributes, providing insights into the role of various health indicators in the likelihood of developing diabetes.

For more details, visit [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data).

## Part I - Data Processing

### Importing the Dataset
Import the library needed and the dataset.
```python
import pandas as pd
dataset = pd.read_csv('diabetes.csv')
```

Show the details of the dataset.
```python
dataset.head(10)
```
![image](https://github.com/user-attachments/assets/bccae72f-2066-43d1-af22-c5b9d01046cd)


Then, inspect the dataset.
```python
dataset.info()
```
<img src="https://github.com/user-attachments/assets/bb02b1e9-6a63-46bc-924d-51612930c69b" width="450" height="300">

Next is the description of the dataset. This is to see what are the minimum and maximum values of the features in the dataset.
```python
dataset.describe()
```
![image](https://github.com/user-attachments/assets/eed7b599-0f95-455b-8951-d88fd6a4ba0f)

Check if there are columns that have missing values.
```python
dataset.isna().sum()
```
<img src="https://github.com/user-attachments/assets/7787fb00-bc7d-4691-beda-1ccfc6872ef1" width="250" height="200">


Show what are the features or columns in the dataset.
```python
dataset.columns
```
<img src="https://github.com/user-attachments/assets/a5d78238-1cdf-4ca4-bdef-7b9a04f264de" width="600" height="65">

### Data Cleaning
Most of the features have 0 values which seems impossible in real life. That is why data cleaning is performed. Here, replace zeros with the mean of each column where zero is implausible.

```python
dataset["Glucose"] = dataset["Glucose"].replace(0, dataset["Glucose"].mean())
dataset["BloodPressure"] = dataset["BloodPressure"].replace(0, dataset["BloodPressure"].mean())
dataset["SkinThickness"] = dataset["SkinThickness"].replace(0, dataset["SkinThickness"].mean())
dataset["Insulin"] = dataset["Insulin"].replace(0, dataset["Insulin"].mean())
dataset["BMI"] = dataset["BMI"].replace(0, dataset["BMI"].mean())
dataset
```
![image](https://github.com/user-attachments/assets/c3b0c646-561a-4a8a-9137-4b27ba828dd4)

Thwn, display the dataset again to check if there's still features who have 0 values on them.
```python
dataset.describe()
```
![image](https://github.com/user-attachments/assets/1a777edb-3ea7-4339-b756-947bd490c2ad)

### Getting the Inputs and Output

Assign for the Inputs and Output values. Inputs being all the features aside from the Outcome, and Outcome as the Output value.

```python
X = dataset.drop(["Outcome"], axis = 1)
y = dataset["Outcome"]
```

Then, display the X and y values.
```python
x
```
![image](https://github.com/user-attachments/assets/ca08b9c9-91ff-4f93-af99-149e26564dc2)

```python
y
```
<img src="https://github.com/user-attachments/assets/7d222b0d-73ba-4eff-b938-ea29d720ec1f" width="300" height="200">

### Creating the Training Set and the Test Set

Here, I used the ratio of 85-15 for the Train Test Split. 
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
```
Then, display the X_train, X_test, y_train, and y_test.
```python
X_train
```
![image](https://github.com/user-attachments/assets/9c26a95b-d99c-4107-a4f7-0be8338bb7f1)

```python
X_test
```
![image](https://github.com/user-attachments/assets/f6892f38-6d50-4191-9ff6-d68501b3377f)

```python
y_train
```
<img src="https://github.com/user-attachments/assets/0511f75d-73d8-409b-92ca-ee48d7467018" width="300" height="200">

```python
y_test
```
<img src="https://github.com/user-attachments/assets/378cd00e-6747-424b-82ff-f996ca32ed22" width="300" height="200">

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
```
```python
X_train
```
<img src="https://github.com/user-attachments/assets/378cd00e-6747-424b-82ff-f996ca32ed22" width="300" height="200">

## Part 2 - Building and Training the Model
### Building the Model
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
```

### Training the Model
```python
model.fit(X_train, y_train)
```
<img src="https://github.com/user-attachments/assets/ef83801d-81fa-4712-b266-d0193625e10f" width="200" height="50">

### Inference
```python
y_pred = model.predict(sc.transform(X_test))
y_pred
```
<img src="https://github.com/user-attachments/assets/c757cf24-853e-4b4d-ae12-1f6bec2cfc74" width="500" height="100">

Making the prediction of a single data point with:

1. Pregnancies = 10
2. Glucose = 130
3. BloodPressure = 70
4. SkinThickness = 70
5. Insulin = 80
6. BMI = 25
7. DiabetesPedigreeFunction = 1
8. Age = 21

```python
model.predict(sc.transform([[10,130,70,70,80,25,1,21]]))
```
<img src="https://github.com/user-attachments/assets/15e0afd4-2dd3-4498-8ec3-eb42d5b90329" width="80" height="25">

## Part 3: Evaluating the Model
### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```
<img src="https://github.com/user-attachments/assets/aa267d58-4f9e-48ca-ad4d-d4c0d663beba" width="125" height="40">

```python
cm =  confusion_matrix(y_test, y_pred)
cm
```
<img src="https://github.com/user-attachments/assets/aa267d58-4f9e-48ca-ad4d-d4c0d663beba" width="125" height="40">

```python
import seaborn as sns
sns.heatmap(cm, annot = True)
```
![image](https://github.com/user-attachments/assets/e0079923-4e5d-4646-83f0-435df810da67)

```python
from sklearn.metrics import classification_report
classification_report(y_test, y_pred)
```
<img src="https://github.com/user-attachments/assets/0094883e-fc8c-489b-a5ee-89d06fc35639" width="350" height="150">

### Accuracy
This is to check the accuracy score of using this model.
```python
(71+25)/(71+25+7+13)
```
<img src="https://github.com/user-attachments/assets/50861402-27b8-44a5-8e8f-74db1c299e19" width="150" height="25">

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
```
<img src="https://github.com/user-attachments/assets/50861402-27b8-44a5-8e8f-74db1c299e19" width="150" height="25">


# Conclusion

The recorded accuracy for the Logistic Regression model is 82.75%, indicating that this model is fairly reliable for predicting whether an individual has diabetes using the Pima Indians Diabetes Dataset. However, there are other models that achieve higher accuracy than the Logistic Regression model, with some reaching over 90%. These include the Bagging Decision Tree model, Random Forest, Support Vector Machine (SVM), Gaussian Naive Bayes, and others. Additional data cleaning and imputation could improve accuracy, though excessive cleaning lowered the Logistic Regression model’s performance in this case. As a result, I opted to replace only the zero values in certain features, as it’s unlikely for such values to be valid given the dataset’s features. 

