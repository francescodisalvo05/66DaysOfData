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
> * LinkedIn #0 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-yfinance-ta-activity-6770455330833891329-62hu

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
> * LinkedIn #1 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-finance-machinelearning-activity-6770807300945936384-pT7t

`DAY-2`: Today I finished the first chapter of De Prado's book and I implemented some financial data structure thanks to the proposed exercises. I appreciated the importance of sampling, in fact the author pointed out that one of the most common errors regard the information used for training a model. The remaining data structures regards **Information-Driven bars**. Their purpose is to sample more frequently when we gain new information. Yesterday we saw Tick bars, Volume bars and Dollar bars. Today I discovered Tick imbalanced bars, Volume imbalanced bars and Dollar imbalanced bars. The main difference is that in this case we sample when we exceed a pre-defined expectation. 

Finally, in order to check what I studied so far, I solved the first exercise of the chapter and I took the opportunity for learning the fundamentals of plotly, a fairly well known library that allows to create interactive charts!

> * Book : [Advances In Financial Machine Learning](https://www.amazon.it/Advances-Financial-Machine-Learning-Marcos/dp/1119482089/ref=sr_1_1?__mk_it_IT=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=ZMLKR6L4EISG&dchild=1&keywords=advances+in+financial+machine+learning&qid=1614284766&sprefix=advances+in+fin%2Caps%2C198&sr=8-1)
> * LinkedIn #2 : https://www.linkedin.com/posts/francescodisalvo-pa_day2-activity-6771197741126172672-jaYg

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day2.png" height="400px"/>
</p>

`DAY-3`: Few days ago I was wondering how to deal with huge quantity of data (in the order of GBs). So, Today I saw an interesting YouTube video made by DecisionForest regarding the memory optimization in pandas. The first issue in Pandas is that it uses by default "int64" or "float64", therefore, a first improvement could be made by downcasting these features into "int32" and "int32" (or even 16). By using this approach he was able to save around 40% of memory on a huge dataset. 

The second step is to save the dataframe into some other optimized format. He proposed "parquet", an Apache Hadoop’s columnar storage format, but I also discovered many others on the following blogpost that I would like to test much more in detail!

> * Youtube video: https://www.youtube.com/watch?v=-cLPasRzJeY&t
> * Blogpost : https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d
> * LinkedIn #3 : https://www.linkedin.com/posts/francescodisalvo-pa_how-to-reduce-memory-usage-and-loading-time-activity-6771545540988542976-_24s

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
> * LinkedIn #4 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-machinelearning-datascience-activity-6771882444967694336-u7nd

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
> * LinkedIn #5 : https://www.linkedin.com/posts/francescodisalvo-pa_day5-activity-6772237509658189824-5OgY

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
> * LinkedIn #6 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-boosting-bagging-activity-6772612009772118020-VoF1

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
> * LinkedIn #7 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-machinelearning-datascience-activity-6772945304846118913-T01L

`DAY-8`: Today I played with Streamlit, a Python library that allows to easily create a web applications in a very simple way! 
Think that this entire script took me just 52 lines. Streamlit also allows to host it on their servers for free, upon request.

I also spent some time figuring out some cool features I can implement along the way. Any idea or feedback is always well received!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day8.jpg" height="500px"/>
</p>


> * Video : https://user-images.githubusercontent.com/66080706/110019754-1c7cbc80-7d29-11eb-9a25-37fe2fbbd8ac.mp4
> * Code : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day8.py
> * LinkedIn #8 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-finance-activity-6773330996357021696-JxTs

`DAY-9`: Today I was really busy, so I had a brief look at some possible technical indicators that could be implmented in python. 

Here is the "top 7", according to Investopidia : https://www.investopedia.com/top-7-technical-analysis-tools-4773275

Then, I came up with some interesting features to implement in the next days and I'll surely post here all the future updates!

> * LinkedIn #9 : https://www.linkedin.com/posts/francescodisalvo-pa_top-7-technical-analysis-tools-activity-6773715324526174208-9FFN

`DAY-10`: Today I implemented the Moving Average, a technical analysis tool that tries to cut out the noise from the trend, by updating the average price in a given period. 

In theory, if the price is above the MA the trend is up, and vice versa. The window of the moving average strongly depends on the trader's time horizon, but a common range goes from 10 to 200.  

It seems to be a very well known approach and I also discovered two different strategies that I will probably implement tomorrow!

Of course it is not perfect, for two main reasons:
1. the future is unpredictable by nature (Taleb docet),
2. it does not work well with volatile stocks (e.g. cryptocurrencies).

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day10.png" height="300px"/>
</p>

> * Source : https://www.investopedia.com/top-7-technical-analysis-tools-4773275
> * LinkedIn #10: https://www.linkedin.com/posts/francescodisalvo-pa_day-10-activity-6774030583296233472-4EE6

`DAY-11`: Today I implemented the SMA (Simple Moving Average), EMA (Exponential Moving Average) and I also tested three associated strategies.

Ih short: 
- SMA : a technical analysis tool that tries to cut out the noise from the trend, by updating the average price in a given period. In theory, if the price is above the MA the trend is up, and vice versa.
- EMA : the idea is the similar to the previous one, but the exponential moving average gives more importance (in terms of weights) to the most recent observations.

The first strategy that I implemented is the "Simple Crossover", that uses just on single SMA. It tells us that when the price crosses above or below a moving average to signal a potential change in trend.

The second one uses two different SMAs, one for the long period and one for the short period. When the shorter-term MA crosses above the longer-term MA, it’s a buy signal and vice versa, when the shorter-term MA crosses below the longer-term MA, it’s a sell signal.

Finally, the third one is the same as the previous one, but instead of using two SMAs, it uses two EMAs.

I cannot quantify which is the best one (and actually I think that no one is the best one, at all), but I should make some more tests!

I should consider some thresholds for the signals, because as you can see from the graphs, there are short periods where I receive too much signals. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day11.png" height="300px"/>
</p>

> * SMA : https://www.investopedia.com/top-7-technical-analysis-tools-4773275
> * EMA : https://www.investopedia.com/terms/e/ema.asp
> * LinkedIn #11 : https://www.linkedin.com/posts/francescodisalvo-pa_day11-activity-6774418286193987584-HbSl

`DAY-12`: Today I implemented my first naive backtester!

I used a super basic approach but I am quite satisfied. I just considered an initial amount that would be entirely invested at each "buy signal" and rebalanced at each "sell signal", considering the adjustement given by the ratio "sell_price / buy_price". I didn't consider any additional cost as fees, final taxes and so on. 

Then, yesterday I realized that in some points I had too many signals, sometimes also impossible (e.g. sell,sell,..). In order to fix that, I just used a flag "pos" that was equal to 1 if I got a position (so I would be able to sell but not to buy anymore) and 0 otherwise.  Then, I added these logical constraints to the main strategy and nothing more!

I tested the Crossover strategy with two EMAs (15 and 100 days) on 4 stocks with a symbolic amount of 10k (each) from '2018-03-06'. I obtained the following results (net profts):
* Tesla (TSLA) : + 58022.05
* Facebook (FB) : + 1579.55
* Google (GOOGLE) : + 8545.52
* Gamestop (GME) : + 215050.11

They seems impressive but I might have done some mistakes somewehere, so do not take these values as gold! 
Looking at the trends and signals, they seems promising, but of course I tested it in a very few stocks. There would be much other stocks where I would lose a lot of money! 


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day12.jpg" height="300px"/>
</p>

> * LinkedIn #12 : https://www.linkedin.com/posts/francescodisalvo-pa_day12-activity-6774766883167051776-Q-aj

`DAY-13`: Today I improved my naive-backtester. 

I didn't take that long, but the result is much more clear now! I took the opportunity to test it on Twitter with an initial amount of $10,000. 

The result is quite interesting because from March 2019 to February 2019 I lost around $3,000. Then, the profit started increasing and it is still holding the position, with a current profit of $7,776.00. Of course we should see when it would have been sold! 

So, I noticed with my own eyes the limits of this strategy. When the stock is highly volatile, this strategy tends to not understand the various corrections, so the results are not so good. On the other hand, it performed well in the long term (also because the stock has experienced a remarkable rise).


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day13.png" height="400px"/>
</p>

> * LinkedIn #13 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-algotrading-datascience-activity-6775105830531407872-Gusq

`DAY-14`: 

Today I started looking for some new technical analysis indicators and I went a bit deeper on the RSI (Relative Strength Index). It measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.

Its range of values goes from 0 to 100. Typically, when it goes below 30, it generally indicate that the stock is oversold, whereas it goes above 70, it means that it is overbought.

I'm planning to implement a couple of other indicators and then I would like to move on some "Machine Learning oriented" tasks applied to the financial sector.

> * LinkedIn #14 : https://www.linkedin.com/posts/francescodisalvo-pa_relative-strength-index-rsi-activity-6775513168534872064-gEa8