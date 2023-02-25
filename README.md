# Pima Indian Diabetes Prediction

## Introduction
For our study we will use the Pima Indian dataset, initially collected by the National "Institute of Diabetes and Digestive and Kidney Diseases". Several constraints were placed on selecting these instances from a more extensive database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. The datasets consist of several medical predictor variables and one target variable, Outcome. Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Data Preparation
The dataset contains a large number of missing values which have been saved as the value 0 as shown to the right. Using this will mislead the model. Hence, we will replace the missing values with their medians.
Tts clearly visible from the correlation table below that the Glucose attribute is very important in the outcome prediction as its correlation coefficient is comparatively high at 0.495 followed by BMI, skin thickness, age, pregnancies, blood pressure, diabetes pedigree function and insulin.

Skin thickness seems to have a high correlation of 0.57 with BMI and so we drop skin thickness in order to avoid correlation between attributes. Pregnancies also seem to be highly correlated with the age attribute so we drop this as well in order to prevent the effects of multicollinearity on the machine learning algorithm.

Insulin and blood pressure seems to be very loosely correlated to the outcome so we drop that too. From this, we find that Glucose, BMI, Age and Diabetes Pedigree Function are the most important attributes and hence our Machine learning models will use only these.

To see the distribution of all attributes, we plotted violin plots separately for positive and negative classes. The data for most attributes, except for blood pressure does not seem to follow a normal distribution and hence will require to be centred and scaled.

## Libraries used
The Pandas library was implemented for the purpose of loading the CSV data as a DataFrame and manipulating the data. The NumPy library was used to make calculations and manipulate the data arrays. Matplotlib and Seaborn for plotting. The sklearn library was the most extensively used right from the StandardScaler and pipeline method to using GridSearchCV for hyper-parameter selection and the deployment of all the ML models and implementing their metrics.

## Classification Methods
- Logistic regression
    - Parameters selected:
        - Solver : ['liblinear', 'lbfgs']
        - Penalty : [None, 'l1', 'l2']
        - max_iterations : [15, 20, 30, 40]
- Decision Trees
    - Parameters selected:
        - criterion :['gini', 'entropy']
        - splitter: ['best', 'random']
        - max_depth : [2,4,5,6,7]
        - min_samples_split : [2,3,4,5,6]
        - min_samples_leaf: [1,2,3,4,5]
        - max_features': [None, 'auto', 'sqrt', 'log2']
- K nearest neighbours
    - Parameters used:
        - n_neighbors: [5, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        - algorithm : ['auto', 'ball_tree', 'kd_tree', 'brute']
        - weights : ['uniform', 'distance']

GridSearchCV was used to find the optimal parameters for all the above
## Training and Testing Process
We split the cleaned dataset into training and testing data using sklearn's train_test_split method. The training data had about 80% of the shuffled data points, while the test data made up 20% of the dataset. For each algorithm, we used GridSearchCV to select the best model parameter using 10 folds. The model with the highest 10-fold average validation accuracy was selected and compared to the training accuracy to ensure that the model is not overfitting or underfitting the data. Once confirmed, the trained model was used to make predictions on the previously unseen test data, and the accuracy, recall, precision, f1-score, and confusion matrix were displayed.

## Conclusion
Logistic regression is the more acceptable method among the three because it takes less computational power and has a much higher testing accuracy of nearly 77% and also a higher F1 score compared to the K-nn and Decision tree models which indicates that this model has a similar precision and recall, unlike the others where the difference between the two is a bit higher. Although the training accuracies are higher for the decision trees and K-nn models there is a small amount of disparity between the training and validation accuracies (~2%) indicating slight overfitting which is evident from its lower but not too far off accuracy values on the never before seen test data.

# Pima Indians Diabetes Dataset Classification


##  How to Open The project
This project includes a Python notebook file (ipynb), Dataset CSV file and a pdf report.
inorder to run ipynb file we need to run it as a Jupyter notebook and change the location of of the dataset in
'data = pd.read_csv(r'-----'.
