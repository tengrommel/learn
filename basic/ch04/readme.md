# Feature Selection and Feature Engineering
> Feature selection - also known as variable selection, attribute selection, or variable subset selection - is a method used to select a subset of features(variables, dimensions) from an initial dataset.

Feature selection is a key step in the process of building machine learning models and can have a huge impact on the performance of a model. Using correct and relevant features as the input to your model can also reduce the chance of overfitting, because having more relevant features reduces the opportunity of a model to use noisy features that don't add signal as input. Lastly, having less input features decreases the amount of time that it will take to train a model. 

Feature selection is important because it can:
- Shorten training times
- Simplify models and make them easier to interpret
- Enhances testing set performance by reducing overfitting

One important reason to drop features is the high correlation and redundancy between input variables or the irrelevancy of certain features. These input variables can thus be removed without incurring much loss of information. Redundant and irrelevancy are two distinct notions, since one relevant feature may be redundant in the presence of another relevant feature with which it is strongly correlated.

Feature engineering in some ways is the opposite of feature selection. With feature selection, you remove variables. In feature engineering, you create new variables to enhance the model. In many cases, you are using domain knowledge for the enhancement.

Feature selection and feature engineering is an important component of your machine learning pipeline, and that's why a whole chapter is devoted to this topic.

By the end of this chapter, you will know:

- How to decide if a feature should be dropped from a dataset
- Learn about the concepts of collinearity, correlation, and causation
- Understand the concept of feature engineering and how it differs from feature selection
- Learn about the difference between manual feature engineering and automated feature engineering. When is it appropriate to use each one?

# Feature selection
> In the previous chapter, we explored the components of a machine learning pipeline. A critical component of the pipeline is deciding which features will be used as inputs to the model.

For many models, a small subset of the input variables provide the lion's share of the predictive ability. In most datasets, it is common for a few features to be responsible for the majority of the information signal and the the rest of the features are just mostly noise.

It is important to lower the amount of input features for a variety of reasons including:

- Reducing the multi collinearity of the input features will make the machine learning model parameters easier to interpret. Multicollinearity(also collinearity) is a phenomenon observed with features in a dataset where one predictor feature in a regression model can be linearly predicted from the other's features with a substantial degree of accuracy.

- Reducing the time required to run the model and the amount of storage space the model needs will allow us to run more variations of the models leading to quicker and better results.

- The smaller number of input features a model requires, the easier it is to explain it. When the number of features goes up, the explainability of the model goes down. Reducing the amount of input features also makes it easier to visualize the data when reduced to low dimensions(for example, 2D or 3D).

- As the number of dimensions increases, the possible configurations increase exponentially, and the number of configurations covered by an observation decreases. As you have more features to describe your target, you might be able to describe the data more precisely, but your model will not generalize with new data points - your model will overfit the data. This is known as the curse of dimensionality.

Let's think about this intuitively by going through an example. There is a real estate site in the USA that allows real estate agents and home owners to list homes for rent or for sale. Zillow is famous, among other things, for its Zestimate. The Zestimate is an estimated price using machine learning. It is the price that Zillow estimates a home will sell for if it was put on the market today. The Zestimates are constantly updated and recalculated. How does Zillow come up with this number? If you want to learn more about it, there was a competition on Kaggle that has great resources on the Zestimate. You can find out more here:

    https://www.kaggle.com/c/zillow-prize-1
    
The exact details of the Zestimate algorithm are proprietary, but we can make some assumptions. We will now start to explore how we can come up with our own Zestimate. Let's come up with a list of potential input variables for our machine learning model and the reasons why they might be valuable:

- Square footage: Intuitively, the bigger the home, the more expensive it will be.
- Number of bedrooms: More rooms, more cost.
- Number of bathrooms: Bedrooms need bathrooms.
- Mortgage interest rates: If rates are low, that makes mortgage payments lower, which means potential homeowners can afford a more expensive home. 
- Year built: In general, newer homes are typically more expensive than older homes. Older homes normally need more repairs.
- Property taxes: If property taxes are high, that will increase the monthly payments and homeowners will only be able to afford a less expensive home.
- House color: At first glance, this might not seem like a relevant variable, but what if the home is painted lime green?
- Zip code: Location, location, location.
- Comparable sales: One of the metrics that is commonly used by appraisers and real estate agents to value a home is to look for similar properties to the "subject" property that have been recently sold or at least are listed for sale, to see what the sale price was or what the listing price currently is.
- Tax assessment: Property taxes are calculated based on what the county currently thinks the property is worth. This is publicly accessible information.

These could all potentially be variables that have high predictive power, but intuitively we can probably assume that square footage, the number of bedrooms, and number of bathrooms are highly correlated. Also intuitively, square footage provides more precision than the number of bedrooms or the number bathrooms and keep the square footage and don't lose much accuracy. Indeed, we could potentially increase the accuracy, by reducing the noise.

Furthermore, we can most likely drop the house color without losing precision.

Features that can be dropped without impacting the model's precision significantly fall into two categories:

- Redundant: This is a feature that is highly correlated to other input features and therefore does not add much new information to the signal.

- Irrelevant: This is a feature that has a low correlation with the target feature and for that reason provides more noise than signal

One way to find out if our assumptions are correct is to train our model with and without our assumptions and see what produces the better results. We could use this method with every single feature, but in cases where we have a high number of features the possible number of combinations can escalate quickly.

Let's analyze three approaches that are commonly used to obtain these insights.

# Feature importance
> The importance of each feature of a dataset can be established by using this method.

Feature importance provides a score for each feature in a dataset. A higher score means the feature has more importance or relation to the output feature. 

Feature importance is normally an inbuilt class that comes with Tree-Based Classifiers. In the following example, we use the Extra Tree Classifier to determine the top five features in a dataset:
    
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier
    import numpy as np
    import matplotlib.pyplot as plt
    data = pd.read_csv("train.csv")
    X = data.iloc[:,0:20] 
    # independent columns
    y = data.iloc[:,-1] 
    # pick last column for the target feature
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class
    # feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X. columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()

# Univariate selection
统计测试可以用来确定哪些特征与输出变量具有最强的相关性。 scikit-learn库具有一个名为SelectKBest的类，该类提供一组统计测试以选择数据集中的K个“最佳”功能。

以下是对非负特征使用卡方统计检验以选择输入数据集中五个最佳特征的示例：

    
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    data = pd.read_csv("train.csv")
    X = data.iloc[:,0:20] #independent columns
    y = data.iloc[:,-1] #pick last column for the target feature
    #apply SelectKBest class to extract top 5 best features
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    scores = pd.concat([dfcolumns,dfscores],axis=1)
    scores.columns = ['specs','score']
    print(scores.nlargest(5,'score')) #print the 5 best features

# Correlation heatmaps

当特征的不同值之间存在关系时，两个特征之间存在关联。例如，如果房价随着平方英尺的增加而上涨，则这两个特征被认为是正相关的。可能存在不同程度的相关性。 如果一个特征相对于另一个特征一致地改变，则这些特征被认为是高度相关的。相关性可以是正的（特征的一个值的增加会增加目标变量的值）或负的（特征的一个值的增加会降低目标变量的值）。

correlation是介于-1和1之间的连续值。
- 如果两个变量之间的相关性为1，则存在完美的直接相关性。 
- 如果两个特征之间的相关性为-1，则存在完美的逆相关性。 
- 如果两个要素之间的相关性为0，则两个要素之间没有相关性。

通过热图，可以轻松地确定哪些功能与目标变量最相关。我们将使用以下代码使用seaborn库绘制相关要素的热图：

    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    data = pd.read_csv("train.csv")
    X = data.iloc[:,0:20] #independent columns
    y = data.iloc[:,-1] # pick last column for the target feature
    #get the correlations of each feature in the dataset
    correlation_matrix = data.corr()
    top_corr_features = correlation_matrix.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

存在更正式和不太直观的方法来自动选择特征。存在许多这些方法，并且在scikit-learn软件包中实现了许多方法。接下来提到对这些方法进行分类的一种方法。
