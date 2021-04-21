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

`DAY-15`: 

Today I implemented the RSI, that I mentioned yesterday. 

The last one that I'll implement will be the MACD (Moving Average Convergence/Divergence)! Then, I'll use streamlit for building a dashboard with all these technical analysis indicators that I coded so far.

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day15.png" height="400px"/>
</p>

> * LinkedIn #15 : https://www.linkedin.com/posts/francescodisalvo-pa_day15-activity-6775866282802737152-x4F7
> * https://it.wikipedia.org/wiki/Relative_Strength_Index
> * https://www.investopedia.com/terms/r/rsi.asp

`DAY-16`:  Today I implemented the Moving Average Convergence Divergence (MACD), another techincal analysis indicator defined as the differnce among two different Exponential Moving Averages (EMAs), a long-term one (26 periods) and a short-term one (12). Finally, the signal line is defined as a 9-EMA on the MACD values. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day16.png" height="400px"/>
</p>

> * LinkedIn #16 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-algotrading-datascience-activity-6776254545782677504-1jI- 
> * https://www.investopedia.com/terms/m/macd.asp

`DAY-17`:  Today I spent some time doing some sketches for my "stock dashboard". Then, I watched an amazing conference by Leda Braga, CEO of Systematica Investments.


> * LinkedIn #17 : https://www.linkedin.com/posts/francescodisalvo-pa_leda-braga-data-science-and-its-role-in-activity-6776611878455005184-N7-4 
> * https://www.youtube.com/watch?v=9WDO8sqiy_Y&t=18s

`DAY-18`: Today I started building the frontend of "Stock Manager", where I will try to show (and to sum up) everything I did in the past weeks. You can already see the skeleton!

I still need to familiarize with Streamlit because I realized that there are a lot of hidden features. I would like to make it as versatile as possible, because I am planning to add two (or more) other sections, so I will need to manage a multi-page layout. 

![main-·-Streamlit-–-Mozilla-Firefox-_Navigazione-anonima_-2021-03-14-21-12-01](https://user-images.githubusercontent.com/66080706/111082808-79872800-850a-11eb-8954-8e6341d52790.gif)

> * LinkedIn #18 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-finance-algotrading-activity-6776963630178025472-dGGL

`DAY-19`: Today I continued working on my stock manager. I added some general informations about the stock, as suggested me yesterday. Then I also organized and cleaned the code, in order to generalize a bit better all the features that I have in mind. I realized that I am not so organized as I thought, so I will try to improve also in this way!

Finally, I started playing with the RSI indicator and I implemented the associated trading strategy. Unfortunately, I would have lost $ 129.24 on the Apple's stock!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day19.jpg" height="400px"/>
</p>

> * LinkedIn #19 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-finance-algotrading-activity-6777332702480744448-AXtesssss

`DAY-20`: Today I implemented the trading strategy based on the MACD indicator (Moving Average Convergence/Divergence). This time I was luckier and I would have earned $922. 

This was my 4th implemented indicator and I think I won't test other strategies or indicators later on. I have learned some technical indicators and I had a lot of fun,  but I think that I won't ever put my money in any automatic trading strategy (at least with my current knowledge).

In the following days I'll try to complete the current dashboard with some "frontend tricks" in order to conclude this first short milestone! 

I'm still figuring out my next steps: in particular I would like to study a bit deeper NLP and/or Forecasting. They would be both useful for the finance sector anyway.


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day20.jpg" height="400px"/>
</p>

> * LinkedIn #20 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6777661385410142208-7aDc

`DAY-21`: Today I decided to start focusing on NLP and I spent some time looking for some good resources!

In particular I checked the Stanford course on NLP with Deep Learning (thanks to Dave Emmanuel Magno) and also the fantastic repository of Thinam Tamang on his #66DaysOfData in NLP: https://lnkd.in/dQwUCSZ

I decided to start with "Natural Language Processing with Python" by S. Bird, E. Klein, and E. Loper in order to learn the basis of NLP and I will decide the next one along the way!

> * LinkedIn #21 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-activity-6778058136348581888-1VxU

`DAY-22`: Today I approached the Natural Language Processing from scratch. Even if I have already a bit of experience for some academic projects, I preferred to make a cleaned swap and to start from the bases. So, I started with tokenization and a text preprocessing.

The tokenization is the process by which the initial string is splitted into smaller units (called tokens). These tokens can be words, digits or punctuation. Then, in order to avoid  useless tokens they can be easily filtering with regular expressions or by iterating and checking the given conditions.

> * LinkedIn #22 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-nlp-activity-6778383127241867264-ia_x
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_22.ipynb

`DAY-23`: Today I explored the difference among "Stemming" and "Lemmatization".

**Stemming** is the process of producing morphological variants of a root/base word. For example [likes,liked,liking'] become all 'like'. It is extremely useful for reducing the number of distinct words and for retrieving more precise information. There are several algorithms, the most common ones are the **porter** and the **snowball**. The porter's stemmer is based on the idea that the suffixes in English are made up of a combination of simpler suffixes. Then, the snowball stemmer is considered the evolution of the porter's one, because it can be also used for non-english words!

**Lemmatization** is the process of grouping together different inflected words. It is similar to stemming, but here we obtain meaningful words. It is more precise, but in contrary, the computation is slower.

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day23-1.jpg" height="400px"/>
</p>

> * LinkedIn #23 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-nlp-activity-6778762019068768256-cP3p
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_23.ipynb
> * Picture : https://www.kaggle.com/general/185500

`DAY-24`: Today I faced the "tf-idf" and the "tf-df", two statistics computed for each token.

The **term frequency inverse document frequency** (tf-idf) gives an higher importance to all the tokens that occur frequently in a single document but rarely on the entire collection. So, it is suitable for heterogeneous documents.

Then, the **term frequency document frequency** (tf-df) gives an higher importance to the tokens that are more frequent over the entire collection, so it is suitable for homogeneous documents.

> * LinkedIn #24 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-nlp-activity-6779146951867805696-Uuu-
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_24.ipynb

`DAY-25`: Today I went through "topic scores", a way for clustering documents by their meaning.

The Latent semantic analysis (LSA) is an algorithm that analyze the relationship of the words, in order to cluster them into topics. Since the number of topics is (obviously) much smaller than the number of topics, it is commonly used for reduce the dimension of your initial matrix. A slightly different algorithm is Linear Discriminant Analysis (LDA), which breaks down a document into a single topic.

LDA is one of the fastest algorithms for dimension reduction, however, it is a supervised algorithm, so it requires some initial labels.

In the picture below there is the implementation of the algorithm, proposed in the book "Natural Language Processing in Action".

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day25.jpg" height="400px"/>
</p>

> * LinkedIn #25 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6779462174587809792-1VJR
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_25.ipynb
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python

`DAY-26`: Today I went through the Latent Semantic Analysis.

Latent Semantic Analysis (LSA), as I mentioned yesterday, is based on the well known Singular Value Decomposition (SVD). It is possible to truncate the tf-idf matrix in order to drastically reduce the dimension of the problem. This new representation highlights the "latent sentiment" of these topics. So, the LSA tells you which dimensions are relevant to the semantic of the documents. In fact the "low variance" topics may represents just noise. 

Then, I tried to implement a simple pipeline with 'A Million News Headlines' dataset'. Firtsly, I cleaned the data by removing digits and hashtags (if any) I splitted each headline in tokens. Then, I decided to normalize the tokens with the Stemming technique because it is faster than the Lemmatization and finally, I used the TruncatedSVD for clustering these headlines in "topics" (7 in this case). Thanks to a snippet that I found on a Kaggle notebook I was able to get the top 10 elements for each cluster (topic).

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day26.jpg" height="400px"/>
</p>

> * LinkedIn #26 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6779867210631401472-wlGA
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_26.ipynb
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python
> * Kaggle notebook : https://www.kaggle.com/rcushen/topic-modelling-with-lsa-and-lda

`DAY-27`: Today I needed to recover some lectures for my university, so I couldn't go further with my NLP studies. 

Therefore, I saw a youtube video by Cassie Kozyrkov. Her short videos are always extremely interesting! I like how she compares the "learning theory" as a teacher that tries to teach to his students.

> * Linkedin #27 : https://www.linkedin.com/posts/francescodisalvo-pa_mfml-019-how-to-avoid-machine-learning-activity-6780232245660659712-PXb5
> * Youtube : https://www.youtube.com/watch?v=ZQtuTqmr4WI

`DAY-28`: Today I went through the Latent Dirichlet Allocation.

The **Latent Dirichlet Allocation** (LDiA) is a generative probabilistic model for collections of discrete data. It assumes that each document is a linear combination of a given number of topics, each one represented by a distribution of words. Once defined the ldia vector on a dataset of "spam/not spam" messages, I trained an LDA (Linear Discriminant Analysis) classifier.

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day28.jpg" height="400px"/>
</p>

> * Linkedin #28 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6780584093328527361-JpIr
> * Notebook : https://github.com/francescodisalvo05/66DaysOfData/blob/main/notebooks-scripts/day_28.ipynb
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python

`DAY-29`: Today I completed the first part of "Natural Language Processing in Action" and I spent some time looking for financial blog/websites to scrape, in order to analyze the impact of news on Stock Prices.

In particular I am planning to scrape from finviz.com that for each stock presents a table with its news from several different sources. At the beginning I will extract just the titles, but I might try to implement some "scraping" patterns for some rources (e.g. Yahoo Finance, Bloomberg and so on).

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day29.jpg" height="400px"/>
</p>

> * Linkedin #29 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-finance-machinelearning-activity-6780960306752557056-jJKO
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python

`DAY-30`: Today as I anticipated yesterday I scraped Tesla's financial news from finviz.com.

In particular, for each stock there is a "table" with the headline of the latest news. So, thanks to BeautifulSoup I was able to scrape the main table with id="news-table", then I iterated over each row and I extracted the title, the resource and the link of the full description. Then, I preprocessed the text by removing digits, punctuation and by normalizing with the stemming technique.

In the following days I will try to analyze the correlation among the "sentiment" over all headlines and the stock's trend. 

Feedbacks and ideas are always well received!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day30-1.jpg" height="400px"/>
</p>

> * Linkedin #30 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-webscraping-activity-6781339744325386240-dgvs


`DAY-31` : Today I started reading the first pages of the 6th chapter from "Natural Language Processing in action", regarding Word2Vec. It learns the meaning of the words by simply processing a large corpus of unlabeled data. 

It looks really promising, tomorrow I'll try to complete this chapter and to start implementing it into my small project.

> * Linkedin #31 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6781691298165350400-acbY
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python

`DAY-32` : Today I studied the Word2Vec a bit more in detail, from "Natural Language Processing in Action".

Then, I tried to apply the Word2Vec on the financial news scraped from finviz.com. Unfortunately, I realized that I didn't have enough data (because the headline proposed by finviz are just a small sample). So, I decided to start implement it on a "more complete" dataset. In particular I went through a dataset that I found on kaggle with 1.6 millions tweets.

I spent some time trying to clean it and I realized once and for all that I need to improve with Regular Expressions! After that, I obtained interesting results. Now I need to figure it out how can I implement an unserpvised sentiment analysis. I think I should use KMeans with two clusters, but then I probably need to merge all the weights for each word (?).

As always, feedbacks, ideas and protips are always well received!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day32.jpg" height="300px"/>
</p>

> * Linkedin #32 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6782001756592209920-NHkv
> * Book : Natural Language Processing in Action: Understanding, Analyzing, and Generating Text With Python 
> * Dataset : https://www.kaggle.com/kazanova/sentiment140
> * Hyperparams Word2Vec : https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

`DAY-33` : Today I tried to implement an Unsupervised Sentiment Analysis with Word2Vec, KMeans and further computations. 
Thanks to Sören Grannemann I came across an interesting article that followed my initial idea. Unfortunately I wasn't able to replicate it into my dataset. I struggled for around one hour but then I decided to look for new articles and resources.

The most popular "unsupervised tool" for doing Sentiment Analysis seems to be Vader (Valence Aware Dictionary for sEntiment Reasoning). I was a bit skeptical at the beginning, but to be honest it surprised me. In the picture down below you can see some tests. 

Have you ever used it for your projects? 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day33.jpg" height="300px"/>
</p>

> * Linkedin #33 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6782391314806779904-IVPA
> * Dataset : https://www.kaggle.com/kazanova/sentiment140
> * Medium article : https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483

`DAY-34` : Today I spent a lot of time with Twitter API's in order to find out a way for overcoming the limitation imposed (100 tweets) but I got millions of errors. 
In particular I tried to iterate on a fixed time window as I saw in an article (1) but I face the same error any time: 

{'start_time': ['2021-03-30T18:18Z']}, 'message': "Invalid 'start_time':'2021-03-30T18:18Z'. 'start_time' must be a minimum of 10 seconds prior to the request time."}

I also tried with tweepy but it doesn't work since the new API's update and twitterscraper doesn't return me anything. 

If you have experience with it, I would be glad to receive some tips!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day33.jpg" height="300px"/>
</p>

> * Linkedin #34 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6782755516616581120-AsqP
> * Article : https://towardsdatascience.com/sentiment-analysis-for-stock-price-prediction-in-python-bed40c65d178

`DAY-35` : Today I finally managed Twitter APIs, so I was able to scrape one week of tweets with a given query. 

So, after cleaning the text (removing "#", tags, links and so on) I used VADER for an unsupervised sentiment analysis. In order to analyze the correlation among the sentiment and the stock price I used yfinance for extracting the prices in the same time window.

Then, I grouped the tweets per day and I considered the percentage of positive tweets per day and I plot them with Plotly. This model is far from "consistent" but I guess it is a nice starting point. An important point to consider is that I extracted 4372 tweets, so it doesn't reflect the "popular opinion". In the following days I would like to improve the sentiment analysis, in order to get a slightly better result. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day35.png" height="300px"/>
</p>

> * Linkedin #35 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6783117279832887296-EPGh
> * Article : https://towardsdatascience.com/sentiment-analysis-for-stock-price-prediction-in-python-bed40c65d178

`DAY-36` : Today I tried to improved the Sentiment Analysis "score", even if since it is unsupervised, it cannot easily judged.

Yesterday I used Vader and today I compared it with two other unservised algorithms: TextBlob and Flair.

TextBlob provides the "sentiment" property, which returns the tuple (polarity,subjectiviy) where polarity is a float value between [-1.0,1]. Then, Flair is also able to predict the "sentiment" with a given confidence percentage. 

Among the three models, the most consistent (in this specific sample) seems to be Vader or TextBlob, but it is not so easy to evaluate. The bottlneck of this approach are the Twitter APIs because of its limit. In fact, there is a limit of 100 tweets for each request and it is not possible to send too many requests, in fact after a while I received the Error 429 (Too Many Requests). I'd try with a sleep between each request.

As always, feedbacks are always welcome!


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day36.png" height="300px"/>
</p>

> * Linkedin #36 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6783480099070070784-qqmG
> * Article : https://towardsdatascience.com/sentiment-analysis-for-stock-price-prediction-in-python-bed40c65d178

`DAY-36` : Today I have read that yesterday Elon Musk tweeted: 

"SpaceX is going to put a literal Dogecoin on the literal moon". 

It is well known that Elon Musk "pushed" Dogecoing for a while thanks to his twitter account. So I decided to quantify how much influence he actually has on this cryptocurrency. So I decided to scrape Elon Musk's profile by using selenium in order to collect as much tweets as I could with the keyowrds ['doge','dogecoin','dog','shiba']. Furthermore, I was able to collect 15 tweets, from "2021-02-04" to "2021-04-01" (yesterday).

Finally, I took the "doge-usd" prices from "yfinance", and the result is impressive! This is the reason why I love data science: for any question you have, if you find the right data, you'll be able to answer it (at least, most of the time)!

The curious thing is that he influenced both positively and negatively, in fact if you see three of the most significant falls, his tweet were:
* Feb 11 =  Frodo was the underdoge, All thought he would fail, Himself most of all.
* Feb 14 =  If major Dogecoin holders sell most of their coins, it will get my full support. Too much concentration is the only real issue imo.
* Mar 14 =  I’m getting a Shiba Inu #resistanceisfutile

Even if it doesn't seem so complicate, I am always enthusiastic about these insights! I have also some ideas to implement in the following days from this base.


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day37.jpg" height="300px"/>
</p>

> * Linkedin #37 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6783833851585171456-JcWa
> * Scraping inspiration : https://github.com/israel-dryer/Twitter-Scraper

`DAY-38` : Today I decided that I will slow down a bit this weekend, so I did a wery well review of Gradient Descent thanks to the amazing StatQuest's video. I do not deny that most of the time I prefer to review some topics from this channel due to all the hypnotic intros!

> * Linkedin #38 : https://www.linkedin.com/posts/francescodisalvo-pa_gradient-descent-step-by-step-activity-6784216154496868352-nQPc
> * Gradient Descent : https://www.youtube.com/watch?v=sDv4f4s2SB8

`DAY-39` : Today I briefly read the first chapter (introduction) of "Natural Language Processing with PyTorch".

Since I wanted to practise on PyTorch due to a future project for my "Machine Learning and Deep Learning" course at univerisity, I decided to switch book! The book "Natural Language Processing in action" is very nice book but let's see how it will go with the new one!

> * Linkedin #39 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-activity-6784467599200673793-uLms
> * Book : [Natural Language Processing with PyTorch](https://www.amazon.it/Natural-Language-Processing-Pytorch-Applications/dp/1491978236/ref=sr_1_5?dchild=1&keywords=NLP+with+PyTorch&qid=1617542825&sr=8-5)

`DAY-40` : Today I went through the third chapter of "Natural Language Processing with PyTorch".

First impressions of this book are good. This chapter was focused on the basics of Neural Networks. So, it covered concepts as Perceptron, Activation and Loss functions and finally it explained the neural network "workload", followed by a supervised sentiment analysis in PyTorch (that you can find below).

> * Linkedin #40 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-machinelearning-datascience-activity-6784781781607436288-rw1a
> * Book : [Natural Language Processing with PyTorch](https://www.amazon.it/Natural-Language-Processing-Pytorch-Applications/dp/1491978236/ref=sr_1_5?dchild=1&keywords=NLP+with+PyTorch&qid=1617542825&sr=8-5)
> * Sentiment Analysis : https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_3/3_5_Classifying_Yelp_Review_Sentiment.ipynb

`DAY-41` : Today I went ahead with the book "Natural Language Processing with PyTorch".

I implemented the Multilayer Perceptron (MLP) in PyTorch and I went through the proposed "Surname Classification" example. So, I started implementing, along the book's guidance, the classes Vocabulary, Vectorizer and SurnameDataset. 

Vocabulary coordinates two Python dictionaries, that form a bijection between tokens and integers. Vectorizer, instead, converts individual tokens into integers. Finally SurnameDataset is responsible of the initial data management. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day41.jpg" height="350px"/>
</p>

> * Linkedin #41 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-nlp-machinelearning-activity-6785134262254362624-Qo3G
> * Book : [Natural Language Processing with PyTorch](https://www.amazon.it/Natural-Language-Processing-Pytorch-Applications/dp/1491978236/ref=sr_1_5?dchild=1&keywords=NLP+with+PyTorch&qid=1617542825&sr=8-5)

`DAY-42` : Today I completed the Surname Classification example from "Natural Language Processing with PyTorch". I also started looking at the introduction of Word Embeddings and I will continue it tomorrow!

> * Linkedin #42 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6785638065399447552-U2fd
> * Book : [Natural Language Processing with PyTorch](https://www.amazon.it/Natural-Language-Processing-Pytorch-Applications/dp/1491978236/ref=sr_1_5?dchild=1&keywords=NLP+with+PyTorch&qid=1617542825&sr=8-5)

`DAY-43` : Today I went deeper on Word Embedding with the book "Natural Language Processing with PyTorch".

Word Embeddings methods map large representative vectors into lower dimensional space, maintaining any kind of semantic relationship. The main advantage are: faster computation avoid redundant representation avoid curse of dimensionality representations learned from task specific are optimal

All word embedding methods train with unlabeled data but use some auxiliary supervised tasks in order to extract some correlations among the words, such as “predict the next word”, “predict the missing word in a sequence” and so on.

In the following example I predicted the analogy of a sequence of terms in the order of "word1 : word2 = word3 : X". Here it is possible to see which are the consequences of biased data used to train the algorithms!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day43.jpg" height="450px"/>
</p>

> * Linkedin #43 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-activity-6785984223477014528-CEdW
> * Book : [Natural Language Processing with PyTorch](https://www.amazon.it/Natural-Language-Processing-Pytorch-Applications/dp/1491978236/ref=sr_1_5?dchild=1&keywords=NLP+with+PyTorch&qid=1617542825&sr=8-5)

`DAY-44` : Today I continued what I started yesterday, implementing a script for plotting the word vectors in a 3-dimensional space!

This representation was possible thanks to the t-SNE (t-distributed stochastic neighbor embedding), a manifold learning technique that constructs the probability distribution of the data in the initial dimensional space and then, it tries to maintain such a probability distribution over the smallest dimensional space. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day44.png" height="350px"/>
</p>

> * Linkedin #44: https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-machinelearning-activity-6786380850222129153-qCQJ

`DAY-45` : Today I went through the LSTM paper and a couple of additional resources.

RNNs, due to its construction, suffer to short term memory. In order to mitigate this problem, the LSTM (Long Short Term Memory) architecture proposes to use an additional component, in order to learn long term dependencies. The computation will be heavier due to an increasing in complexity, so this architecture is suggested when you expect to learn long sequences to learn. 

This architecture has three gates : the input gate, the remember (or don't forget) gate and the output gate. Each layer will have its non linear function (sigmoid)
that will determine the outcome. 

So, the final output will be determined by the last sigmoid function, so we might have two possibilities: 0 or 1. 
On the other side, it is possible to controll the memory status in three possible ways:
* reset : input = memory = 0,
* keep : input = 0, memory = 1,
* write : input = 1, memory = 0

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day45.jpg" height="350px"/>
</p>

> * Linkedin #45 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-nlp-deeplearning-activity-6786656052239724545-NOg4
> * Paper : http://www.bioinf.jku.at/publications/older/2604.pdf
> * Picture + Content : https://www.youtube.com/watch?v=5KSGNomPJTE&t=3477s
> * Blogpost : https://colah.github.io/posts/2015-08-Understanding-LSTMs/

`DAY-46` : Today I went through the "Sequence to Sequence Learning" paper. 

This paper proposes an end to end approach to sequence learning that makes minimal assumptions to the sequence structure. It uses a multilayered LSTM to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM decode the target sequence from the vector.

DNNs can perform very well when inputs and targets can be sensibly encoded with vectors of fixed dimensionality but can perform dramatically worse with sequence of tokens. So, the idea is to use LSTM to read the input sequence, one timestamp at a time, to obtain large fixed dimensional vector representation, and then, use another LSTM to extract the output sequence from this former vector.

The traditional approach supposes to have input and output sequence vector with an equal length, and it might be a sort of "bottleneck". In this case the idea to use a double LSTM in order to encode the input sequence to a fixed vector, and then, to map this target vector into the actual prediction. The LSTM architecture has been proposed due to its "larger" tolerance to long term dependencies.

Another interesting trick regards the encoding, in fact the authors proposed to reverse the tokens. So, if "a,b,c" gives "x,y,z", now "c,b,a" gives "x,y,a". This trick allows to have a stronger communication between the "a" and "x".  

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day46.jpg" height="350px"/>
</p>

> * Linkedin #46 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-deeplearning-nlp-activity-6787046839205011456-oWSu
> * Paper : https://arxiv.org/pdf/1409.3215.pdf

`DAY-47` : Today I went through the "Quasi Recurrent Neural Network" paper. 

RNNs are really valuable for modeling sequences, but the main problem is that they're slow, they work with one token at each timestamp. On the other hand, CNN are really valuable for extracting spatial features extremely fast (thanks to the parallelization).

So, the authors propose to alternate convolutional layers and recurrent pooling functions applied in parallel across channels. 

Each layer presents two subcomponents, comparable to the convolution and pooling layers in CNNs. The convolutional one allows parallel computations across mini batches and spatial dimension, whereas the pooling component performs convolutions across minibatch and feature dimensions.

The pooling component captures the most relevant features obtained so far. In this case, the authors  were inspired by the nature of the LSTM and the pooling layers simply mix the hidden states across timestamps, indipendently on each channel of the state vector.

This architecture is opened to severl improvements, such as regularizations, densely connected layers, encoder-decoder structures and so on. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day47.jpg"/>
</p>

> * Linkedin #47 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-nlp-deeplearning-activity-6787421470663446528-HJbn
> * Paper : https://arxiv.org/pdf/1611.01576v1

`DAY-48` : Today I came across an interesting article related to #Leetcode. 

It is often argued the relevance of leetcode for technical interviews and I admit that sometimes I asked myself the usual question : why should I spend time doing DSA exercises instead of improving my Machine Learning skills or going ahead personal projects?

I recognize the fundamental importance of Data Structure and Algorithms, but at the moment I can't say that I am pro or against leetcoding for technical roles.

I'd be very happy to know some other opinions from different point of views! What do you think about this kind of training for technical roles? 

> * Linkedin #48 : https://www.linkedin.com/posts/francescodisalvo-pa_five-things-i-have-learned-after-solving-activity-6787822687193468928-WkLa
> * Article : https://towardsdatascience.com/five-things-i-have-learned-after-solving-500-leetcode-questions-b794c152f7a1

`DAY-49` : Today I studied the flavours of "Federated Learning" thanks to various papers and articles.

Federating Learning was proposed for the first time by Google in 2017 and it was (and it still is) an innovating approach for training machine learning algorithms on clients' devices (smartphone et simila). 

The main idea is that the smartphone downloads the current model and it improves by learning the data from the smartphone itself, and just a small amount of data will be sent to the server for improving the general model. So in this way all the training data will be stored just on the client. 

The challenges of this interesting technique are related to:
* computational resources of the devices
* bandwith and latency 
* non iid data distributions
* privacy
* fault tolerance

The training process is determined by the federated averaging algorithm, where the main server chooses a subset of data clients for a training step. So, these clients receive the model will send some information in order to improve the model. Finally the main server will optimize the model by averaging all these parameters. In order to preserve the privacy, there will be injected some noise before entering in the main system, so it would be hard to deanonimize them by aggregating them with other data. 

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day49.jpg"/>
</p>

> * Linkedin #49 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-activity-6788179132493643776-Uzpg
> * Article : https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
> * Paper : https://ieeexplore.ieee.org/document/9084352
> * Paper : http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

`DAY-50` : Today, for an academic assignement (in Computer Vision), I studied the paper "Unsupervised Pixel–Level Domain Adaptation" with Generative Adversarial Networks.

The goal of domain adaptation is to train data on a source dataset and apply it on a target dataset generated from a different distribution. The term "unsupervised" is given by the fact that we have no labels in the target domain. 

So, since creating dataset such as ImageNet and COCO is quite expansive, so a feasible alternative is the use of synthetic data for modeling by using unsupervised domain adaptation. So we aim to transfer knowledge learned from a source domain, for which we have labeled data, to a target domain for which we have no labels. 

This was just the flavour of the paper, tomorrow I will go deeper into the model!


<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day50.jpg"/>
</p>

> * Linkedin #50 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-deeplearning-gan-activity-6788574631071621120-Bem-
> * Paper : https://arxiv.org/pdf/1612.05424.pdf 

`DAY-51` : Today I went deeper on the paper "Unsupervised Pixel–Level Domain Adaptation" with Generative Adversarial Networks.

Recall that the goal of unsupervised domain adaptation is to train data on a source dataset and apply it on a target dataset generated from a different distribution where we have no labels. 

As you can see from the image, there are three main components on the model: the gnerator (G), the Discriminator (D) and the task specific classifier (T). 
In short, the generator G generates an image from a synthetic image and a noise vector. Then, the discriminator D discriminates among real and fake images whereas the task-specific classifier T, instead, assign a labels to the fake image. 

This model provides several benefits such as:
- Decoupling from the Task-Specific Architecture
- Generalization Across Label Spaces
- Training Stability
- Data Augmentation
- Interpretability

But also, it outperforms previous state of the art unsupervised domain adaptation techniques on several different image datasets!

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day51.jpg"/>
</p>

> * Linkedin #51 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-deeplearning-cv-activity-6788937427907055616-ypzu
> * Paper : https://arxiv.org/pdf/1612.05424.pdf 

`DAY-52` : Today I spent most of my free time with Selenium, in order to accomplish a task for my current Omdena's project. 

In particular I was able to dynamically scrape from a table divided in pages. So at each iterations I clicked on the "next" page to scrape. It was a bit frustrating at the beginning, but the result is awesome! 

> * Linkedin #52 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-datascience-webscraping-activity-6789298168015163392-nhr0

`DAY-52` : Today I went through t-SNE (t-distributed stochastic neighbor embedding), a manifold learning technique that constructs the probability distribution of the data in the initial dimensional space and then, it tries to maintain such a probability distribution on a smaller dimensional space.

The source is pretty obvious: StatQuest! 

> * Linkedin #52 : https://www.linkedin.com/posts/francescodisalvo-pa_statquest-t-sne-clearly-explained-activity-6789659579073470464-Dx2K
> * Paper : https://www.youtube.com/watch?v=NEaUSP4YerM

`DAY-53` : Today I went through the PyTorch's implementation of the Pixel-level Domain Adaptation algorithm.

You can find the code here : https://github.com/eriklindernoren/PyTorch-GAN

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day53.jpg"/>
</p>

> * Linkedin #53 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-gan-deeplearning-activity-6790012698496114688-3JoM

`DAY-54` : Today I went through the iCaRL's paper, from Sylvestre-Alvise Rebuff et al. 

iCaRL states for Incremental Classifier and Representation Learning, and it is devoted to a "different" training approach. As you can imagine, this training will be incremental. It means that we periodically fed the model with new class labels to train. 

It presents several advantages: 
- It does not require a sufficient training set before learning;
- It can continuously learn to improve when the system is running;
- It can adapt to changes of the target concept
- It doesn't need a priori informations about the number or distribution of the data

The main issue (that it is argued in the literature) of Incremental Learning is the cathastrophic forgetting, and this paper proposes a new approach for overcoming this issue, based on three main components:
- Distillation loss term: stabilizes output and limits overhead
- Set of exemplars: selection procedure and discard on the fly
- Nearest mean of exemplars classifier : automatic adjustement to representation change. For each new sample, we'll assign it to the nearest approximate class mean in the feature space.

<p align="center">
  <img src="https://github.com/francescodisalvo05/66DaysOfData/blob/main/images/day54.png"/>
</p>

> * Linkedin #54 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-deeplearning-incrementallearning-activity-6790356918868185089-jBBz

`DAY-55` : Today I went through a couple of papers related to Incremental Learning and Semantic Segmentation for an incoming academic projects in Computer Vision.

> * Linkedin #55 : https://www.linkedin.com/posts/francescodisalvo-pa_66daysofdata-computervision-cv-activity-6790726280342990848-qE5g