<h1 align="center"> #66DaysOfData in Financial Machine Learning </h1>

The challenge consists on learning data science every day for 66 days and sharing the progress on a social media ([see more](https://www.youtube.com/watch?v=qV_AlRwhI3I&t=12s)).
<p align="center">
  <img src="https://images.pexels.com/photos/159888/pexels-photo-159888.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" height="500px"/>
</p>


`DAY-0` : Today I discovered two interesting libraries for extracting financial information. The first one is **yFinance** that extracts its information from Yahoo Finance. Thanks to the Ticker module we are able to select one or more tickers. Then, with the **history** method we are allowed to download the historical values in a specific period within a particular interval (daily,weekly..). It provides seven features: Open, High, Low, Close, Volume, Dividends, Stock Splits. Finally, we can gain some other information with the info method (123 in total). These information are related to the geographical location, economical status and so on. 

Then, the second library is **ta** (Technical Analysis) that implements 32 indicators based mostly on the volume, volatility and trend. Understanding all these features would require a huge effort because it is "trading oriented", but I will surely study them later on.

> * https://algotrading101.com/learn/yfinance-guide/ <br />
> * https://technical-analysis-library-in-python.readthedocs.io/en/latest/ <br />
> * https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543 <br />

`DAY-1` : Today I started reading the first pages of "Advances in Financial Machine Learning" by Marcos Lopez de Prado. It is divided into five parts: 
data analysis,modeling,backtesting,useful financial features,high-performance computing recipes. The first chapter of the first part is devoted to **Financial Data Structures**. First thing first, financial data can be found in several different ways and the essential types can be summarized in four macro categories: <br />
* Fundamental data: information mostly obtained by business analytics, for example assets, sales and so on. Due to its variety, it may be hard to manipulate.
* Market data: information regarding all trading activities. Since they are precise and with a very well known structure, they can be easily analyzed and manipulated.
* Analytics: derivative data, based on multiple sources.
* Alternative data: it can be produced by individuals (e.g. social media), business processes and sensors. 

All the information that we gain, must be processed and converted into a machine-friendly version. One of the most common representation is through **bars**.
* Time bars: they are the most common ones and we explored them yesterday with yfinance.
* Tick bars: they sample the time bars based on the trading activity. This allows to avoid taking samples from low trading activity intervals (that would be not so relevant).
* Volume bars: they sample the time bars based on the volume of the trades, without considering just the number, as the tick bars do. 
* Dollar bars: they sample observations every time a market value is exchanged. 

Tomorrow I will focus on some other advanced financial data structure and I will try to conclude the first chapter of this book!

> Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)

`DAY-2`: Today I finished the first chapter of De Prado's book and I implemented some financial data structure thanks to the proposed exercises. I appreciated the importance of sampling, in fact the author pointed out that one of the most common errors regard the information used for training a model. The remaining data structures regards **Information-Driven bars**. Their purpose is to sample more frequently when we gain new information. Yesterday we saw Tick bars, Volume bars and Dollar bars. Today I discovered Tick imbalanced bars, Volume imbalanced bars and Dollar imbalanced bars. The main difference is that in this case we sample when we exceed a pre-defined expectation. 

Finally, in order to check what I studied so far, I solved the first exercise of the chapter and I took the opportunity for learning the fundamentals of plotly, a fairly well known library that allows to create interactive charts!

> Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day2.png" height="400px"/>
</p>

`DAY-3`: Few days ago I was wondering how to deal with huge quantity of data (in the order of GBs). So, Today I saw an interesting YouTube video made by DecisionForest regarding the memory optimization in pandas. The first issue in Pandas is that it uses by default "int64" or "float64", therefore, a first improvement could be made by downcasting these features into "int32" and "int32" (or even 16). By using this approach he was able to save around 40% of memory on a huge dataset. 

The second step is to save the dataframe into some other optimized format. He proposed "parquet", an Apache Hadoop’s columnar storage format, but I also discovered many others on the following blogpost that I would like to test much more in detail!

> * Youtube video: https://www.youtube.com/watch?v=-cLPasRzJeY&t
> * Blogpost : https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d

`DAY-4`: Today I studied the third chapter of "Advances in Financial Machine Learning" that focuses on how to label financial data. 

One of the biggest mistakes is to label observations with fixed thresholds. In fact, we should set the "take profit" and "stop loss" by using function of the risk! A possible way of labeling is with the so called "triple-barrier method", where the aim is to label an observation according to the first barrier touched. There are two horizontal bars (stop loss and take profit) and one vertical bar (expiration limit).

We buy if we first hit the upper barrier, we sell if we first hit the lower one and we may decide to buy or sell if we hit the middle bar. This decision can be influenced by the stock volatility. From the picture below, we can distinguis: (a) starting date, (b) stop-loss, (c) take profit, (d) starting date plus the number of days you are planning to hold it [1]

The previous labeling alone is not so effective, in fact we need to know how much we should bet (bet size). This is called by the author "meta labeling". It helps increasing the f1-score, because we build a first model with an higher recall and then we correct the low precision by the meta labeling approach. In fact, it tries to filter out the false positive!

In short:
1. Apply the triple-barrier method 
2. Generate the meta-labels
3. Use a binary classifier (buy/sell)  on the meta-labels in order to improve the performances

Tomorrow I will try to implement these steps and let's see what I will obtain! 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day4.jpg" height="400px"/>
</p>


> * Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)
> * Figure and [1]: https://towardsdatascience.com/the-triple-barrier-method-251268419dcd
> * Another useful resource : https://ai.plainenglish.io/start-using-better-labels-for-financial-machine-learning-6eeac691e660

`DAY-5` : Today I was approaching the implementation of triple barrier method and meta labeling but I found another interesting topic to focus on: the Bollinger Bands. 

They are a technical analysis tool for generating oversold and overbought signals. In particular they give signals for eventual long or short positions. In a nutshell, you take a "long" position if you believe that the stock will rise up, and on the contrary you take a "short" position if you believe that ste stock value will decrease. 

I took the opportunity to work with the dollar bars (bars indexed by the traded dollar volume) thanks to a new implementation that I found in a very well done article [1]. Then, I implemented a simple script for defining these short or long positions and finally I plotted the result! 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/Day5.png" height="400px"/>
</p>

> * Jupyter notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks/Day5.ipynb
> * [1] article : https://ai.plainenglish.io/start-using-better-labels-for-financial-machine-learning-6eeac691e660
> * Bollinger Bands : https://www.investopedia.com/terms/b/bollingerbands.asp
> * Long and Short positions : https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/stock-purchases-and-sales-long-and 

`DAY-6` : Today I jumped to the 6th chapter of "Advances in Financial Machine Learning" by Marcos Lopez de Prado that covers "Ensemble methods". 

In particular I have dwelt on the difference between "Boosting" and "Bagging". At first reading I haven't fully understood the topic, so I decided to read it from the following very well done article : https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/

Both bagging and boosting are ensemble learning techniques, it means that they combine several base model in order to improve the overall performances. Here's the main features:

* Learners : they both get N learners by generating additional data in the training stage and the training sets are obtained by random sample with replacement. In bagging, any element has the same probability to be considered, whereas in boosting there are weights associated to the sampling process. 
* Weights : in bagging each model is indipendent, whereas in boosting each model takes into account the previous classifiers' success.
* Classificaton : each learner will make his own predictions. The final classification will be given by a simple average in bagging and by a weighted average in boosting. 

So, they both decrease the variance but boosting tries also to construct a stronger model. 

This was just a gentle introduction of the topic, but if you want to know more, I found a very detailed article on TowardsDataScience! 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day6.png" height="200px"/>
</p>

> * Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)
> * https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/
> * https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205

`DAY-7`: Today I studied the 7th chapter of "Advances in Financial Machine Learning" by Marcos Lopez de Prado, devoted to "Cross Validation in Finance". 

I was thinking about K-Fold with finance data for a while, and today De Prado confirmed my concerns. In fact he first pointed out "why k-fold fails in finance". The main reasons are basically two:

* in financial data we cannot leverage on the hypothesis that the observations are **IID** (Independent and Identically Distributed),
* the testing sets are used multiple times in order to develop the model, so we may induct some bias

Of course, since we'll have overlapping observations, we may observe the so called phenomenon of "Data Leakage" (when we use in training data some information that we do not expect during the prediction). 

A first improvement can be made by "purging the training set", so we remove from the training set all observations whose labels are overlapped with the ones contained in the test set. If this approach is not enough for preventing the data leakage, De Prado proposes to impose an embargo on training observations after every test set. The idea is to define an embargo period ( typically 0.01T - where T is the number of bars) after every test set, in which we discard training samples! It is much more clear on the following picture.

I tried to implement it on my own but I got stuck for a while, so I will try again tomorrow! 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day7.jpg" height="300px"/>
</p>

> * Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)
> * Picture and slides : https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257420
> * See more : https://medium.com/@samuel.monnier/cross-validation-tools-for-time-series-ffa1a5a09bf9


