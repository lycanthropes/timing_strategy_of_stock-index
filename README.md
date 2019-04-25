# 50ETF指数择时策略
>>根据深度学习LSTM模型对50ETF期权的VIX指数涨跌的预测结果进行对50ETF指数的择时交易。中国股市的涨跌与期权VIX
   
 呈正相关，因此本策略即在看涨VIX时买进50ETF指数，看跌时卖出50ETF指数。
   <br>
    >>首先，本文使用多层LSTM模型对VIX指数的预测结果如下：

![image](https://github.com/lycanthropes/timing_strategy_of_50ETF_strategy/blob/preview/images/Training_result.png)
>>其次，本策略的净值曲线如下：
![image](https://github.com/lycanthropes/timing_strategy_of_50ETF_strategy/blob/preview/images/money_curve.png)
