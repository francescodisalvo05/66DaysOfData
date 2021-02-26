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
