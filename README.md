# bank_loan_analysis
Using Python to analyze Bank Loan Interest Rate data to build a model in order to predict loan interest rate, given all the other metrics. This data includes metrics such as the loan length, monthly income, loan purpose, and FICO score, etc.
1.Data Cleaning 
There are 2200 instances in the dataset, which means that it is fairly small by Machine Learning standards, so we have to handle the NaN carefully instead of deleting them directly (as seen chart 1). 
Firstly, most Dtypes are object, including amount_requested, interest rate, FICO scores, etc. Transforming these columns to numeric and we found that loan amount requested and amount funded have nearly perfect correlation (as seen graph1), so we dropped columns ‘amount_requested’, as well as ‘ID’ and ‘State’, which is useless for the model. Besides, for ‘FICO.Range’, the values are all range, like 730-734. We added a new column ‘FICO’ and set value equals to the smaller figure of the range since all the range has the same difference value. For ‘emplt_len’, some values are ‘< 1 year’ or ‘10+ years’ (as seen chart 2), we used replace function and set 0.5 instead of less than 1 year and 10 for more than 10 years. 
Secondly, in order to avoid risk of introducing a significant sampling bias, we decide to stratified sampling by the loan length (as seen graph2) to guarantee that the train set is representative of the overall data. And we set 0.2 for test set. After splitting the data by ‘loan_len’, we started to count the null and using imputer function to set median to the numerical attributes. 
Meanwhile, OneHotEncoder class was used to convert categorical values into one-hot vectors for  columns ’home_ownership’ and ‘ loan_purpose’ respectively. Besides, we used Pipeline class to help with such sequences of transformations.
2. Visualize the Data to get Insights
The correlation  between numerical features was shown as chart 3, the data distributions was as shown as graph 3, and pairplot was used  as graph 4 to visualize the relationship between variables clearly.
3.Selection and Training the model
1) LinearRegression
After fitting the linear regression model , the mean squared error is 0.2
2) DecisionTreeRegressor
The mean squared error for this model is around 0.019.
3) RandomForestRegressor
Random Forests work by training many Decision Trees on random subsets of the features, then averaging out their predictions. The mean squared error is around 0.0068, smallest of the three.
4) Grid Search
Scikit-Learn’s GridSearchCV uses cross-validation to evaluate all the possible combinations of hyperparameter values. According to the result, the best estimator: max_features are 20, and the relative importance of each attribute for making accuate predictions was displayed in chart 4. 
4. Evaluate System on the Test Set
Since  RandomForestRegressor performed best, so we chose this model and evaluate it on the test set. The mean squared error is around 0.018. 
