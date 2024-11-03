<p>Data Processing<p/>
 
* Dataset file have been cleaned and changed to reflect to the data.
* Missing values were found and reflected.
* Columns have been categorized.
* The numbers have been normalized.

<p>Model Implementation<p/>
 
*  pandas
*  scikitlearn
*  numpy 
  
<p>Evaluation Metrics<p/>
 
*  R-Squared = 0.8774370925197137
*  Adjusted R-Squared = 0.8197604301760495
*  Mean Squared Error = 7.652119236368388

<p>Interpretation<p/>

*  The intercept represents the expected value of the dependent variable when all independent variables are equal to zero. In this context, it indicates that if all predictor variables were zero, the response variable would be approximately 93.45.
*  A negative coefficient indicates that as the corresponding predictor variable increases by one unit, the response variable is expected to decrease, assuming all other variables remain constant.
*  A positive coefficient indicates that as the corresponding predictor variable increases, the response variable is expected to increase.
*  The intercept and coefficients give us valuable insights into how each predictor affects the response variable. To really evaluate the model's accuracy and the importance of each predictor, it's crucial to look at their statistical significance along with the coefficients.
  
## Reference
### Linear Regression Dataset
The **Boston House Prices** dataset contains information about various factors influencing housing prices in Boston suburbs, collected by the U.S. Census Service. It includes 506 observations with 13 features, such as the average number of rooms, crime rate, accessibility to highways, and more, all used to predict the median value of homes. This dataset is commonly used in linear regression analysis to understand how these factors impact housing prices.

You can explore it further [here](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices).
