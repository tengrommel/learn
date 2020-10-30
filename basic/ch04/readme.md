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


# Wrapper-based methods

使用包装器方法时，使用以下步骤将特征选择的问题实质上简化为搜索问题：

- 1. 功能子集用于训练模型
- 2. 根据每次迭代的结果，将要素添加到子集中或从子集中删除

包装方法通常计算的成本是很高的。以下是一些示例：

  - 正向选择：正向选择方法是一个迭代过程，首先要在数据集中没有特征。在每次迭代过程中，都会添加功能以改善模型的性能。如果性能得到改善，则功能将保留。无法改善结果的功能将被丢弃。该过程一直持续到模型停止改进为止。
  - 向后消除：使用向后消除方法时，所有特征最初都存在于数据集中。重要性最低的要素将在每次迭代过程中删除，然后，该过程将检查模型的性能是否得到改善。重复该过程，直到没有观察到明显的改善为止。
  - 递归特征消除：递归特征消除方法是一种贪婪的优化算法，其目标是找到性能最佳的特征子集。它迭代创建模型，并在每次迭代期间存储性能最佳或最差的功能。它将使用其余功能构造下一个模型，直到所有功能用尽。然后根据特征的消除顺序对其进行排序。

# Filter-based methods

指定度量标准，并基于该度量标准过滤特征。基于过滤器的方法的示例包括：
- Pearson's Correlation: 该算法用于量化两个连续变量X和Y之间的线性相关性。其值可以介于-1到+1之间。
• Linear discriminant analysis (LDA): LDA可用于查找特征的线性组合，这些特征描述或分隔了分类变量的两个或多个级别（或类）。
• Analysis of Variance (ANOVA): 方差分析类似于LDA，不同之处在于它使用一个或多个类别自变量和一个连续因变量来计算。它提供了统计检验，以了解几组的平均值是否相等。
• Chi-Square: 卡方（Chi-Square）是一种统计检验，应用于分类变量组，以使用其频率分布确定它们之间相关或关联的可能性。

要记住的一个问题是基于过滤器的方法不会消除多重共线性。因此，在根据输入数据创建模型之前，必须执行其他处理以处理特征多重共线性。


# Embedded methods
> 嵌入式方法使用带有内置特征选择方法的算法。嵌入式方法结合了filter和wrapper方法的优点。通常使用带有内置特征选择方法的算法来实现它们。

两种流行的嵌入式方法实现如下：
- Lasso regression：执行L1正则化，这会增加等于系数幅值的绝对值的惩罚
- Ridge regression: 它执行L2正则化，这将增加等于系数幅度平方的惩罚

这些算法也很常用：
    - Memetic algorithm
    - Random multinomial logit
    - Regularized trees

到此结束本章的第一部分。您应该准备好应对自己的功能选择项目。现在，准备好进入特征工程领域。特征选择与我们如何减少变量数量以提高准确性有关。特征工程则相反。它问：如何创建新变量以使我们的模型更具性能？

# Feature engineering

就像明智而系统地选择特征可以通过删除特征使模型更快，性能更高一样，特征工程可以通过添加新特征来实现相同目的。乍看之下这似乎是矛盾的，但是要添加的功能不是由功能选择过程删除的功能。要添加的要素是可能未包含在初始数据集中的要素。 您可能拥有世界上最强大，设计最完善的机器学习算法，但是如果您的输入功能不相关，您将永远无法产生有用的结果。让我们分析几个简单的例子以获得一些直觉。

在上一章中，我们探讨了贷款违约问题。凭直觉，可以推测，如果借款人的薪水高，借款人的违约率就会降低。

同样，我们可以假设，与余额较低的人相比，信用卡余额较大的借款人可能很难偿还这些余额。

现在，我们有了新知识，让我们尝试直观地确定谁将偿还贷款，谁不偿还贷款。如果借款人A的信用卡余额为10,000美元，借款人B的余额为20,000美元，您认为谁有更大的机会还清债务？如果没有其他信息，我们可以说借款人A是一个更安全的选择。现在，如果我告诉您借款人A的年收入为20,000美元，借款人B的年收入为100,000美元，该怎么办？ 那改变了一切，不是吗？我们如何定量地捕捉两个特征之间的关系？银行经常使用所谓的债务收入比（DTI），其计算方法与您想象的一样：
    
            DTI = Debt/Income

因此，借款人A的DTI为0.50，借款人B的债务收入比为0.20。换句话说，借款人A的债务是其债务的两倍，借款人B的债务是其债务的5倍。借款人B有更多偿还债务的余地。我们当然可以在这些借款人的配置文件中添加其他功能，这将改变其配置文件的构成，但是希望这有助于说明功能工程的概念。

更正式地讲，特征工程是数据科学家或计算机生成可增强机器学习模型的预测能力的特征的过程。特征工程是机器学习中的一个基本概念，可能既困难又昂贵。许多数据科学家想直接跳入模型选择，但是辨别哪些新功能将增强模型的能力是一项关键技能，可能需要花费数年才能掌握。

更好的特征工程算法的发展目前正在紧锣密鼓地进行，有一天，在特征工程决策方面，这些特征可能会比高级数据科学家更好，但是，在接下来的几年中，我们预测优秀的数据科学家仍会需求。

特征工程过程可以描述如下：
    - 1. 集体讨论哪些功能相关
    - 2. 确定哪些功能可以改善模型性能
    - 3. 创建新功能
    - 4. 确定新功能是否会增加模型性能；如果没有，放下它们
    - 5. 返回第1步，直到模型的性能达到预期
    
正如我们在示例中看到的那样，拥有领域知识并熟悉数据集对于要素工程很有用。但是，也有一些通用的数据科学技术可应用于数据准备和功能工程步骤中，而不管其域是什么。让我们花一些时间分析这些技术。

我们将探索的技术是：
    - Imputation
    - Outlier management
    - One-hot encoding
    - Log transform 
    - Scaling
    - Date manipulation

# Imputation

数据集“肮脏”且不完美的情况并不少见。缺少值的行是一个常见问题。值丢失的原因可能有很多：

- Inconsistent datasets
- Clerical error
- Privacy issues

不管出于何种原因，缺少值都会影响模型的性能，并且在某些情况下，由于某些算法不会善意对待丢失的值，因此可能会导致停止运行。有多种技术可以处理缺失值。它们包括：

删除缺少值的行 
>这种技术会降低模型的性能，因为它减少了模型必须训练的数据点的数量。

数值插补
> 数值插补只是将缺失的值替换为另一个“有意义的”值。

对于数字变量，这些是常见的替换：
- 使用零作为替换值是一种选择
- 计算整个数据集的平均值，然后用平均值替换缺失值
- 计算整个数据集的中位数，然后用中位数替换缺失值

通常最好使用平均值而不是平均值，因为平均值更容易受到异常值的影响。让我们看几个替换示例：

    #Filling all missing values with 0
    data = data.fillna(0)
    #Filling missing values with medians of the columns
    data = data.fillna(data.median())
    print(data)
Categorical Imputation 
>分类变量不包含数字，而是包含类别。 

例如，红色，绿色和黄色。或香蕉，苹果和橙子。因此，平均值和均值不能与分类变量一起使用。常用的技术是用出现最多的值替换所有丢失的值。

在存在许多类别或类别均匀分布的情况下，使用“其他”之类的名称可能会有意义。

# Outlier management
> 房价是一个很好的分析领域，可以理解为什么我们需要特别注意离群值。无论您居住在世界的哪个区域，您附近的大多数房屋都将落入一定范围内，并且将具有某些特征。也许是这样的：

- 1 to 4 bedrooms
- 1 kitchen
- 500 to 3000 square feet
- 1 to 3 bathrooms

2019年美国的平均房价为226,800美元。您可以猜测，这种房屋可能会具有上述某些特征。但是，也可能有一些离群的房子。也许有10或20间卧室的房子。其中一些房屋可能价值一百万或一千万美元，具体取决于这些房屋可能具有的疯狂定制数量。正如您可能想象的那样，这些离群值将影响数据集中的均值，并且将对均值产生更大的影响。因此，鉴于这些房屋的数量不多，最好删除这些离群值，以免影响其他较常见的数据点的预测。让我们看一下一些房屋价值的图表，然后尝试画出两条最佳拟合线：一条去除所有数据，一条去除高价房屋离群值：

如您所见，如果从最佳拟合线的计算中删除异常值，则该线将更准确地预测低价房屋。因此，简单地删除异常值是处理异常值影响的简单而有效的方法。

那么我们如何确定一个值是否是一个离群值并应将其删除？一种常见的方法是删除落在数据集中某个要素值的标准偏差的特定倍数的离群值。用于乘数的常量更多的是一门艺术，而不是一门科学，但是2到4之间的值很常见：

检测和消除异常值的另一种方法是使用百分位。使用这种方法，我们仅假设要素值的一定百分比是离群值。下降的值百分比又是主观的，并且将取决于领域。

# One-hot encoding

单热编码是机器学习中用于特征工程的一种常用技术。某些机器学习算法无法处理分类特征，因此单热编码是一种将这些分类特征转换为数值特征的方法。假设您有一个标记为“状态”的功能，该功能可以采用三个值（红色，绿色或黄色）之一。因为这些值是分类的，所以不存在哪个值更高或更低的概念。我们可以将这些值转换为数值，从而赋予它们这种特性。

# Log transform

对数转换（或对数转换）是常见的要素工程转换。对数转换有助于展平高度偏斜的值。应用日志转换后，数据分布将被标准化。

让我们再看一个例子，再次获得一些直觉。请记住，当您10岁时，看着15岁的男孩和女孩时，他们在想：“他们比我大得多！”现在想想一个50岁的人和另一个55岁的人。在这种情况下，您可能会认为年龄差异并不大。在这两种情况下，年龄差异均为5岁。但是，在第一种情况下，15岁的年龄比10岁的年龄大50％，在第二种情况下，55岁的年龄比50岁的年龄大10％。

如果我们对所有这些数据点应用对数变换，则将这样的幅度差异归一化。
由于幅度差异的归一化，使用对数变换的模型也将减少异常值的影响，并且使用对数变换的模型将变得更加健壮。

# Scaling
> 在许多情况下，数据集中的数字特征在规模上会与其他特征有很大差异。例如，房屋的典型平方英尺数可能是1000到3000平方英尺之间的数字，而房屋中卧室数量的2、3或4可能是更典型的数字。如果我们不理会这些值，那么如果单独使用比例尺较高的要素，则可能会赋予较高的权重。如何解决此问题？

缩放可以解决此问题。应用缩放后，连续特征在范围方面变得可比。并非所有算法都需要标定值（Random Forest浮现在脑海），但是如果未事先对数据集进行标定，则其他算法将产生无意义的结果（例如k近邻或k均值）。现在，我们将探讨两种最常见的缩放方法。

规范化（或minmax规范化）可在0到1之间的固定范围内缩放要素的所有值。

    data = pd.DataFrame({'value':[7,25, -47, 73, 8, 22, 53, -25]})
    data['normalized'] = (data['value'] - data['value'].min()) /
    (data['value'].max() - data['value'].min())
    print(data)

