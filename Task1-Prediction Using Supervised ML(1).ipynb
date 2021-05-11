# Data Science & Business Analytics intern at The Sparks Foundation
### Author: Jahanvi Gaur

# Task 1 : Prediction Using Supervised ML

# ## Problem Statement : 
#### **Predict the percentage of an student based on the no. of study hours.**

* Algorithm Type: Supervised 
* Language Used: Python
* Algorithm Used: Linear Regression 

### What is Linear Regression is ?
#### LinearRegression, in its simplest form, fits a linear model to the data set by adjusting a set of parameters in order to
                       ## make the sum of the squared residuals of the model as small as possible

# 1. Importing necessary libraries

#importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
%matplotlib inline 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 2. Reading dataset Student_Score.csv

#Read the dataset

Student_scores=pd.read_csv(r"C:\Users\DELL\Desktop\Student_Score.csv")
Student_scores

#Checking the size of dataset using shape
# shape: returns a tuple with each index having the number of corresponding elements.

Student_scores.shape


#removing Row/Columns which have NULL values
#Pandas dropna() method allows the user to analyze and drop Rows/Columns with Null values in different ways.

Student_scores=Student_scores.dropna(how='any')


#Viewing top 5 Records

Student_scores.head()


#Sorting the dataset in Ascending order
#sort_values(): Pandas sort_values() function sorts a data frame in Ascending or Descending order of passed Columns.

Student_scores.sort_values(by=['Scores','Hours'],ascending= True, inplace=False)

# 3. Declaring Variable and ploting Scatter plot of Hours vs Scores

#storing the values of Hours and scores in x and y

x=Student_scores['Hours']
y=Student_scores['Scores']

#ploting scatter plot of given dataset

plt.scatter(x,y,label="stars",color="red",marker="o",s=30)
plt.title('Data Distribution',fontsize=18)
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.show()

# 4. Ploting Histogram of Hours and Scores

#ploting histogram of given dataset Student_score

Student_scores.hist()

# 5. Plotting Histogram of Score

# Plotting histogram of 'Score'

range=(0,100)
bins=8
plt.hist(y,bins,range,color='pink', histtype='barstacked',rwidth=0.8)

plt.xlabel('Scores', fontsize= 15)
plt.ylabel('No. of Students', fontsize=15)
plt.title('Score Distribution',fontsize=18)

plt.show()

# 6. Plotting Histogram of Hours

# Plotting histogram of 'Hours'

range=(0,12)
bins=8
plt.hist(x,bins,range,color='yellow', histtype='barstacked',rwidth=0.8)

plt.xlabel('Hours',fontsize=15)
plt.ylabel('No. of Students',fontsize=15)
plt.title('Hours Distribution',fontsize=18)

plt.show()

#As we see in " Data Distribution " Graph that there is a Linear Relationship between study Hours and student Score . 
#So, we can say that Linearity is present in Data and we can apply Linear Regression in it.

# 7. Reshaping the x and y variable 

#reshape the x and y variable and storing it into X and Y respectively.
# .reshape(): Gives a new shape to an array without changing its data.

X=x.values.reshape(-1,1)
Y=y.values.reshape(-1,1)

# 8. Splitting the dataset intoTraining and Testing Data and Performing Regression

#splitting the Dataset 
# Where,
#  1. X and Y : Arrays
#  2. test_sizefloat or int, default=None
#           If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
#           If int, represents the absolute number of test samples. 
#           If None, the value is set to the complement of the train size. If train_size is also None, it will be set to 0.25.
#  3. random_state:Controls the shuffling applied to the data before applying the split. 
#                  Pass an int for reproducible output across multiple function calls
    
x_train ,x_test ,y_train ,y_test = train_test_split(X,Y,test_size=0.2,random_state=69)


#Performing Linear Regression

stud_reg=LinearRegression()
stud_reg.fit(x_train,y_train)
print("------------------------------ Model Training Completed -----------------------------------")


#As model is Trained successfull .
# Now , appling LR equation

lmodel=stud_reg.coef_*X + stud_reg.intercept_

# Plotting the Regression Graph 

plt.scatter(X,Y, color='red')
plt.plot(X,lmodel, c='green', lw=2)
plt.title('After Regression ', fontsize=18)
plt.xlabel('Hours', fontsize=15)
plt.ylabel('Scores', fontsize=15)
plt.show()
#As we see that Regression line is fitting the data in best way possible

# 9. Predicting values with model and comparing with Observed Values/ Actual values

# .predict(): predict the predicted value of data

y_predict = stud_reg.predict(x_test)

# Creating Observed Score DataFrame 

df=pd.DataFrame(y_test,columns=['Observed Scores '])
df
df['Predicted Scores']=y_predict
df

# Ploting test score against predited score 

plt.scatter(y_test,y_predict, color = 'purple')
plt.xlabel('Y_test Data(Expected)',fontsize=15)
plt.ylabel('Y_predict Data(Prediction)',fontsize=15)
plt.show()


# Now what will be the predicted score of student after studying 9.25 hr/day? 
# Predicting score of student if he/she study for 9.25 hr/day

hours=[[9.25, ]]
our_prediction=stud_reg.predict(hours)

print("------------Testing the Model on bases of Given Hours------------ \n----------- HERE WE GET------------")

print("For Given Hours = {}".format(hours))
print("Predicted Score = {}".format(our_prediction[0]))

# 10.Evaluation model

# Calculating Root mean squared error

from sklearn.metrics import mean_squared_error

#mse= mean squared error

mse =  mean_squared_error(y_test,y_predict)

#rmse= root mean squared error

rmse=np.sqrt(mse)
print(" Now,     Mean Square Error->" , mse)
print("          Root Mean Square Error->" , rmse)


# Calculating Mean Absolute Error

from sklearn.metrics import mean_absolute_error

# mae= mean absolute error

mae=mean_absolute_error(y_test,y_predict)
print("Now, Mean Absolute Error->" , mae)

# Calculating R-squared and Adjusted R-sqaured 

R2= stud_reg.score(x_train,y_train)

N=x_train.shape[0]
P=x_train.shape[1]

Adjust_R2 = 1-(1-R2)*(N-1)/(N-P-1)

print('  \n   R-Squared->', R2)
print('  \n   Adjusted R-Squared->', Adjust_R2)
