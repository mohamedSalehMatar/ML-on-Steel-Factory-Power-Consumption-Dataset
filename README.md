# Machine Learning on Steel Factory Power Consumption Dataset.


## 1 Pre-Processing

### 1.1 Dealing with date column:
Date data posed a challenge as it’s not an integer nor string. Dealing with such issue required transforming this column to pandas’ datetime datatype. This process resulted in the formation of 3 new columns (‘Day’, ‘Month’, ‘Year’) and dropping the old ‘Date’ column.

### 1.2 Checking for the existence of duplicate records:
The duplicates can have a great effect on the model accuracy. Therefore the usage of duplicated().sum() function to detect such defect records. Thankfully, no duplicates were found

### 1.3 Checking for the existence of null value records:
As for the duplicates, null isn’t less of a threat on the model results. The usage of the function isnull().sum() was crucial to access the problem. A total of 3,743 values were empty. 

### 1.4 Filling null values in categorical columns:
Both columns of ‘WeekStatus’ and ‘Day_of_week’ has null values. Using fillna(.mode()), Those null values were filled with the mode of their respective column.

### 1.5 Dealing with outliers:
As the name suggests, the outliers deviate from the general distribution of the data. Such elements must be eliminated before any modelling attempt. The interquartile range equation formed some useful criteria about will be considered an outlier record. After that, we came to conclusion that because of how many outliers was out there. Deleting them wasn’t an option as it will leave the model vulnerable to underfitting. So, the most practical option is filling outlier values with the mean value of their respective columns. Instead of executing such operation when dealing with outliers and when dealing with numerical values, much optimum approach was to turn all outliers to null values and then turning all nulls into mean values.

### 1.6 Filling null values in numerical columns:
Over 1,500 numerical null value was detected using isnull().sum(). Using SciKit-Learn library, more specifically SimpleImputer class, all null values were filled using mean strategy.

### 1.7 Dealing with categorical columns:
By using SciKit-Learn library, more specifically LabelEncoder class, both columns of ‘WeekStatus’ and ‘Day_of_week’ where encoded to fit into the numerical requirements of modelling.


## 2 Modelling 

### 2.1 Plotting the Learning Curve:
Data have been split into 80% training, 10% testing and 10% validation using k-fold technique.
 

4 Regression models are used:

•	Linear Regression

•	Polynomial Regression

•	Ridge Regression

•	Lasso Regression

Polynomial Regression is the best model used, with 96% accuracy. Mean Squared Error (MSE) is calculated for training and validation to plot a learning curve over all folds. In Polynomial Regression Model: training, testing and validation data is transformed to polynomial features of degree 2.


## 2.1 Hyperparameter Tuning: 

Hyperparameter tuning is the process of optimizing the settings that control how a machine learning algorithm learns from data. These settings, called hyperparameters, are external to the model and cannot be directly learned from the dataset. They govern various aspects of the learning process, such as the model's complexity, the rate at which it learns, and how much it penalizes large parameter values (regularization). By fine-tuning these hyperparameters, we aim to maximize the algorithm’s performance on a specific dataset. Hyperparameter Tuning gave the best accuracy on the Lasso Regression Model. The objective is to find the optimal value of the hyperparameter alpha, which controls the strength of regularization, resulting in improved model performance. The hyperparameter tuning process for Lasso Regression involved evaluating two alpha values using k-fold cross-validation. The best alpha value was determined based on the lowest validation error. The final model, trained with the optimal alpha, achieved satisfactory performance on both training and testing datasets. When we used it for the model training and testing accuracy was increased, so there is the difference between the model before and after the hyperparameter tuning.
 






