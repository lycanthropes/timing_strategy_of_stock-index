# 50ETF指数择时策略
>>根据深度学习LSTM模型对50ETF期权的VIX指数涨跌的预测结果进行对50ETF指数的择时交易。中国股市的涨跌与
 期权VIX呈正相关，因此本策略即在看涨VIX时买进50ETF指数，看跌时卖出50ETF指数。
   <br>
    >>首先，本文使用多层LSTM模型对VIX指数的预测结果如下：

![image](https://github.com/lycanthropes/timing_strategy_of_50ETF_strategy/blob/preview/images/Training_result.png)
>>其次，本策略的净值曲线如下：
![image](https://github.com/lycanthropes/timing_strategy_of_50ETF_strategy/blob/preview/images/money_curve.png)
>>策略年化收益率仅有5.43%, 夏普比率0.35。这说明策略仍有改进空间：下一步我计划采用与A股股市相关性更强的AVIX指数（Zheng,Jiang and Chen，2017）作为市场的情绪指标（而非本策略使用的VIX指标）；另外就模型本身而言，我计划进一步改进LSTM模型以更好的寻找AVIX与股指之间存在的关系。
