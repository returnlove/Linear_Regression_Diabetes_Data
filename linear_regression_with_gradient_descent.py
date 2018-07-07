import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
#print(diabetes)

diabetes_X = diabetes.data
diabetes_y = diabetes.target
#print(diabetes_X.shape)

diabetes_X_train, diabetes_X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33, random_state=42)

#train

W =  np.random.uniform(low=-0.1, high=0.1, size= 10)  #dimension of 10
b =  0.0#scalar


learning_rate = 0.01
epochs = 100

for i in range(epochs):
    
    #calculate predictions
    y_pred = diabetes_X_train.dot(W) + b
    
    #error
    error = y_train - y_pred
    mean_sqaured_error = np.mean(np.power(error, 2))




'''
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, y_train)

y_pred = model.predict(diabetes_X_test)

print(model.coef_)
print(metrics.mean_squared_error(y_test, y_pred))
'''