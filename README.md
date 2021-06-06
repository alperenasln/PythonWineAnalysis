### PythonWineAnalysis
Wine Dataset Analysis by using Python

##Problem Definition

Problem arises in the need of wine classification. We need to classify the wines according to its ingredients. Because we need to diverse wines to match with the meal that goes along the best.

We decide on three types for the classification system. First one goes well with the meal that includes white meat, second one goes well with the meal that includes red meat, third one goes well with both.

We believe that best way to solve this problem is to use one of the supervised classification models of wines according to its ingredients.

##Data Analysis

These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

The features are depicted as below in order:

Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline

The features are numerical values of chemicals inside of the wine.
All attributes are continuous.
There are no missing values in the dataset.

![image](https://user-images.githubusercontent.com/70862043/120936303-d4967880-c70f-11eb-9cb7-fcbe98c1774b.png)
 
Statistical Descriptions of the Data:

![image](https://user-images.githubusercontent.com/70862043/120936314-da8c5980-c70f-11eb-94e4-4f228c2931bb.png)

 
We believe all the features are essential to classify and should be used for the model training.


##Data Processing

We have used train_test_split method from sklearn.model_selection with default parameters. Therefore, we picked randomly, and it consist of %20 of the total dataset.
As stated, “Data Analysis” part, we do not have any missing values nor categorical features to handle.
We normalized the values due to fact that size of scale between features are way too much.
 
We have used the formula above within the sklearn.preprocessing package.
Model Selection and Training
Selected Models: Logistic Regression, Decision Tree, Gaussian Naïve Bayes

#Logistic Regression:

![image](https://user-images.githubusercontent.com/70862043/120936405-35be4c00-c710-11eb-9038-189b26f91d93.png) 
 
#Decision Tree:

![image](https://user-images.githubusercontent.com/70862043/120936413-42db3b00-c710-11eb-8e15-18c139ad643c.png)

#Gaussian Naïve Bayes

![image](https://user-images.githubusercontent.com/70862043/120936417-48d11c00-c710-11eb-8659-90ec3231f684.png)

We have selected the best result according to accuracy scores. Therefore, we have picked decision tree model.

##Fine-tune Your Model

We have used GridSearchCV to fine tune our model.
Parameters are as given below:
criterion: entropy, gini
max_depth: 3-9
min_samples_leaf: 1-9

![image](https://user-images.githubusercontent.com/70862043/120936433-67cfae00-c710-11eb-8c49-efbf66259a99.png)

 
Accuracy increased from %90.86~ to %93.05~ thanks to fine-tuning step.
 
##Testing

We have used accuracy, precision, recall, F-score, confusion matrix methods from sklearn.metrics to measure the performance.
Results can be seen at below.
![image](https://user-images.githubusercontent.com/70862043/120936444-7322d980-c710-11eb-94ed-24ffece2924b.png)

 
##Summary

We tried different models to classify our dataset. We believe decision tree model outperforms others in this dataset. Therefore, we fine-tuned that and tested again. Consequently, we achieved around %94 accuracy as a best result. 

Dataset Link
