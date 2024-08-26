import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from FinMind.data import DataLoader
from pandas_market_calendars import get_calendar
import numpy as np
# 輸入參數
stockid = input("請輸入股票代碼: ")
startmoney = float(input("請輸入起始金額: "))
start_date = input("請輸入開始日期 (格式: YYYY-MM-DD): ")
end_date = input("請輸入結束日期 (格式: YYYY-MM-DD): ")

# 取得資料
dl = DataLoader()
df = dl.taiwan_stock_daily(stock_id=stockid, start_date=start_date, end_date=end_date)

# 整理資料格式
df = df.rename(columns={"date": "Date"})
df.set_index("Date", inplace=True)
df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))

# backtesting.py 格式
df1 = df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"})
df2 = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "Volume"})

# 取得 KD 值2
df_kd_k, df_kd_d = talib.STOCH(df2['high'], df2['low'], df2['close'], fastk_period=9, slowk_period=3, slowd_period=3)
df1['K'] = df_kd_k
df1['D'] = df_kd_d

# 定義策略類別
class KdCross(Strategy):
    def init(self):
        pass
    
    def next(self):
        if self.data.K[-1] < 20:
            self.buy()
        elif self.data.K[-1] > 80:
            self.sell()
class BollingerBandStrategy(Strategy):
    n = 20  # 布林通道的回望期
    k = 2   # 布林通道的標準差倍數

    def init(self):
        super().init()
        # 計算布林通道
        self.bbands_upper, self.bbands_mid, self.bbands_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.n, nbdevup=self.k, nbdevdn=self.k)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果價格在或低於布林通道下軌，則買入
        if self.data.Close[-1] <= self.bbands_lower[-1]:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 如果價格在或高於布林通道上軌，則賣出
        elif self.data.Close[-1] >= self.bbands_upper[-1]:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class RSIStrategy(Strategy):
    rsi_period = 14  # RSI的計算週期
    overbought_threshold = 70  # 超賣閾值
    oversold_threshold = 30  # 超買閾值

    def init(self):
        super().init()
        # 計算RSI
        self.rsi_series = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果RSI > 70，則賣出
        if self.rsi_series[-1] > self.overbought_threshold:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] RSI > 70，進行賣出操作')

        # 如果RSI < 30，則買入
        elif self.rsi_series[-1] < self.oversold_threshold:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] RSI < 30，進行買入操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class MovingAverageStrategy200buy5sell(Strategy):
    short_ma = 5  # 短期均線
    long_ma = 200  # 長期均線

    def init(self):
        super().init()
        # 計算均線
        self.short_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.short_ma)
        self.long_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.long_ma)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果股價低於60MA，則買入
        if self.data.Close[-1] < self.long_ma_series[-1]:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 如果股價高於5MA，則賣出
        elif self.data.Close[-1] > self.short_ma_series[-1]:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class MovingAverageStrategy60buy5sell(Strategy):
    short_ma = 5  # 短期均線
    long_ma = 60  # 長期均線

    def init(self):
        super().init()
        # 計算均線
        self.short_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.short_ma)
        self.long_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.long_ma)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果股價低於60MA，則買入
        if self.data.Close[-1] < self.long_ma_series[-1]:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 如果股價高於5MA，則賣出
        elif self.data.Close[-1] > self.short_ma_series[-1]:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class MovingAverageStrategy20buy5sell(Strategy):
    short_ma = 5  # 短期均線
    long_ma = 20  # 長期均線

    def init(self):
        super().init()
        # 計算均線
        self.short_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.short_ma)
        self.long_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.long_ma)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果股價低於20MA，則買入
        if self.data.Close[-1] < self.long_ma_series[-1]:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 如果股價高於5MA，則賣出
        elif self.data.Close[-1] > self.short_ma_series[-1]:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class MovingAverageStrategy5buy5sell(Strategy):
    short_ma = 5  # 短期均線
    long_ma = 5  # 長期均線

    def init(self):
        super().init()
        # 計算均線
        self.short_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.short_ma)
        self.long_ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.long_ma)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])

    def next(self):
        # 如果股價低於5MA，則買入
        if self.data.Close[-1] < self.long_ma_series[-1]:
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 如果股價高於5MA，則賣出
        elif self.data.Close[-1] > self.short_ma_series[-1]:
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
class DailyRSIStrategy(Strategy):
    ma_period = 200  # 均線的回望期
    rsi_period = 14  # RSI的回望期
    rsi_oversold = 30  # RSI的超賣區域
    rsi_overbought = 70  # RSI的超買區域

    def init(self):
        super().init()
        # 計算均線
        self.ma_series = self.I(talib.MA, self.data.Close, timeperiod=self.ma_period)
        # 計算RSI
        self.rsi_series = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
        # 記錄進場的價格
        self.entry_price = None

    def next(self):
        if self.data.index[-1] == self.data.index[0]:
            return

        # 條件1: 價格處於200MA均線上
        if self.data.Close[-1] > self.ma_series[-1]:
            # 條件2: RSI下行至超賣區域
            if self.rsi_series[-1] < self.rsi_oversold:
                # 記錄進場價格
                self.entry_price = self.data.Open[-1]
                # 買入
                self.buy()
                self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Open[-1], 'date': self.data.index[-1]}, ignore_index=True)
                #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 賣出條件
        elif self.entry_price is not None:
            # 條件1: RSI上行至40中線的上方
            if self.rsi_series[-1] > 40:
                # 賣出
                self.position.close()
                self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Open[-1], 'date': self.data.index[-1]}, ignore_index=True)
                #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 條件2: RSI一直在40中線下方波動沒有起色，可以選擇在第11根K線的開盤價賣出
            elif len(self.trades_df) > 0 and len(self.trades_df) % 2 == 0 and len(self.data) >= 11:
                self.position.close()
                self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Open[-1], 'date': self.data.index[-1]}, ignore_index=True)
                #print(f'[{self.data.index[-1]}] 進行賣出操作')

                # 清除先前的交易記錄，以確保每一次買入訊號都被處理
                self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
                self.entry_price = None
class MovingAverageCrossoverStrategy(Strategy):
    short_window = 10  # 10日均線窗口期
    long_window = 30   # 30日均線窗口期

    def init(self):
        super().init()
        # 計算10日均線和30日均線
        self.short_ma = self.I(talib.SMA, self.data.Close, timeperiod=self.short_window)
        self.long_ma = self.I(talib.SMA, self.data.Close, timeperiod=self.long_window)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
        # 記錄進場的價格
        self.entry_price = None

    def next(self):
        if self.data.index[-1] == self.data.index[0]:
            return

        # 判斷入場時機
        if crossover(self.short_ma, self.long_ma):
            # 如果10日均線向上突破30日均線，視為入場的信號
            self.entry_price = self.data.Close[-1]
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 判斷出場時機
        elif crossover(self.long_ma, self.short_ma):
            # 如果10日均線向下跌破30日均線，視為出場的信號
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行賣出操作')

            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
            self.entry_price = None
class ModifiedStrategy(Strategy):
    short_window = 50  # 短期均線的窗口期
    long_window = 200  # 長期均線的窗口期
    rsi_period = 14  # RSI的窗口期
    rsi_overbought = 70  # RSI的超買區域
    rsi_oversold = 30  # RSI的超賣區域
    trend_window = 30  # 趨勢線窗口期

    def init(self):
        super().init()
        # 計算短期均線和長期均線
        self.short_ma = self.I(talib.SMA, self.data.Close, timeperiod=self.short_window)
        self.long_ma = self.I(talib.SMA, self.data.Close, timeperiod=self.long_window)
        # 計算相對強弱指標(RSI)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # 計算趨勢線
        self.trend_line = self.I(talib.SMA, self.data.Close, timeperiod=self.trend_window)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
        # 記錄進場的價格
        self.entry_price = None

    def calculate_slope(self, data, window):
        x = np.arange(window)
        y = data[-window:]
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def next(self):
        if self.data.index[-1] == self.data.index[0]:
            return

        # 判斷入場時機
        if crossover(self.short_ma, self.long_ma) and self.calculate_slope(self.short_ma, self.short_window) < self.calculate_slope(self.long_ma, self.long_window):
            # 如果移動平均線的短期均線向下穿越長期均線，並且均線的斜率變得平緩，視為入場的信號
            self.entry_price = self.data.Close[-1]
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 判斷出場時機
        elif self.entry_price is not None:
            # 如果相對強弱指標(RSI)穩定在70以上或30以下，並且持平震盪，視為出場的信號
            if (self.rsi[-1] > self.rsi_overbought or self.rsi[-1] < self.rsi_oversold) and self.calculate_slope(self.rsi, self.rsi_period) == 0:
                # 如果技術指標突破趨勢線，視為出場的信號
                if self.data.Close[-1] > self.trend_line[-1]:
                    self.position.close()
                    self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
                    #print(f'[{self.data.index[-1]}] 進行賣出操作 (趨勢線突破)')

                    # 清除先前的交易記錄，以確保每一次買入訊號都被處理
                    self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
                    self.entry_price = None
class MACDStrategy(Strategy):
    short_window = 12  # 短期EMA窗口期
    long_window = 26  # 長期EMA窗口期
    signal_window = 9  # Signal線窗口期

    def init(self):
        super().init()
        # 計算MACD
        self.macd_line, self.signal_line, _ = self.I(talib.MACD, self.data.Close, fastperiod=self.short_window, slowperiod=self.long_window, signalperiod=self.signal_window)
        # 創建一個 DataFrame 來追蹤交易
        self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
        # 記錄進場的價格
        self.entry_price = None

    def next(self):
        if self.data.index[-1] == self.data.index[0]:
            return

        # 判斷入場時機
        if crossover(self.macd_line, self.signal_line):
            # 當MACD線從下方向上穿越信號線，視為入場的信號
            self.entry_price = self.data.Close[-1]
            self.buy()
            self.trades_df = self.trades_df.append({'action': 'buy', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)
            #print(f'[{self.data.index[-1]}] 進行買入操作')

        # 判斷出場時機
        elif self.macd_line[-1] < self.signal_line[-1] and self.macd_line[-2] >= self.signal_line[-2]:
            # 當MACD線從上方向下穿越信號線，視為出場的信號
            self.position.close()
            self.trades_df = self.trades_df.append({'action': 'sell', 'price': self.data.Close[-1], 'date': self.data.index[-1]}, ignore_index=True)


            # 清除先前的交易記錄，以確保每一次買入訊號都被處理
            self.trades_df = pd.DataFrame(columns=['action', 'price', 'date'])
            self.entry_price = None
class MonthlyInvestmentStrategy(Strategy):
    def init(self):
        self.buy_day = 6  # 每月買入的日期
        self.buy_amount = 5000  # 每次買入的金額
        self.total_investment = 0  # 總投入成本
        self.calendar = get_calendar("XTAI")  # 台灣股市的交易行事曆

    def next(self):
        # 檢查日期是否為交易日
        current_date = self.data.index[-1]
        if len(self.calendar.valid_days(current_date, current_date)) == 0:
            return

        # 每月6號進行買入，並確保總投入未超過起始資金
        if current_date.day == self.buy_day and self.total_investment < startmoney:
            # 計算可以買入的股數
            num_shares = int(self.buy_amount / self.data.Close[-1])
            # 計算實際花費的金額
            actual_cost = num_shares * self.data.Close[-1]
            # 更新總投入成本
            self.total_investment += actual_cost
            # 執行買入
            self.buy(size=num_shares)

        # 在end_date將股票全部賣出
        if current_date == pd.to_datetime(end_date):
            self.position.close()
            print(f'總投入成本：{self.total_investment}')

# 添加其他策略類別...

# 策略選擇
strategies = {
    1: KdCross,
    2: BollingerBandStrategy,
    3: RSIStrategy,
    4: MovingAverageStrategy200buy5sell,
    5: MovingAverageStrategy60buy5sell,
    6: MovingAverageStrategy20buy5sell,
    7: MovingAverageStrategy5buy5sell,
    8: DailyRSIStrategy,
    9: MovingAverageCrossoverStrategy,
    10: ModifiedStrategy,
    11: MACDStrategy,
    12: MonthlyInvestmentStrategy,
}

# 選擇策略
strategy_index = int(input('''請選擇策略 (1-12):\n
1.K<20買入，K>80賣出\n
2.布林通道回測，觸底買入，觸頂賣出\n
3.RSI<30買入，>70賣出\n
4.股價低於200MA，買入，高於5MA，賣出\n
5.股價低於60MA，買入，高於5MA，賣出\n
6.股價低於20MA，買入，高於5MA，賣出\n
7.股價低於5MA，買入，高於5MA，賣出\n
8.MA200均線搭配RSI指標\n
9.MA10跟MA30黃金交叉買入，死亡交叉賣出\n
10.鈍化現象判斷買入或賣出\n
11.MACD黃金交叉買入，死亡交叉賣\n
12.每個月6號固定買入25000，直到資產用盡，設定的最後一個交易日全部出清 "))\n
'''))
selected_strategy = strategies[strategy_index]

# 執行回測
bt = Backtest(df1, selected_strategy, cash=startmoney, commission=.002)
stats = bt.run()
result_str = str(stats)
print(f'股票代號：{stockid}')
print(f'起始資產：{startmoney}')
result_str = result_str.replace('Start', '起始日期')
result_str = result_str.replace('End', '結束日期')
result_str = result_str.replace('Duration', '持續時間')
result_str = result_str.replace('Exposure Time [%]', '曝光時間 [%]')
result_str = result_str.replace('Equity Final [$]', '最終資產 [$]')
result_str = result_str.replace('Equity Peak [$]', '最高資產 [$]')
result_str = result_str.replace('Return [%]', '回報 [%]')
result_str = result_str.replace('Buy & Hold Return [%]', '持有回報 [%]')
result_str = result_str.replace('Return (Ann.) [%]', '回報率 (年化) [%]')
result_str = result_str.replace('Volatility (Ann.) [%]', '波動率 (年化) [%]')
result_str = result_str.replace('Sharpe Ratio', '夏普比率')
result_str = result_str.replace('Sortino Ratio', '索提諾比率')
result_str = result_str.replace('Calmar Ratio', '卡瑪比率')
result_str = result_str.replace('Max. Drawdown [%]', '最大回撤 [%]')
result_str = result_str.replace('Avg. Drawdown [%]', '平均回撤 [%]')
result_str = result_str.replace('Max. Drawdown Duration', '最大回撤持續時間')
result_str = result_str.replace('Avg. Drawdown Duration', '平均回撤持續時間')
result_str = result_str.replace('# Trades', '# 交易數量')
result_str = result_str.replace('Win Rate [%]', '勝率 [%]')
result_str = result_str.replace('Best Trade [%]', '最佳交易 [%]')
result_str = result_str.replace('Worst Trade [%]', '最差交易 [%]')
result_str = result_str.replace('Avg. Trade [%]', '平均交易 [%]')
result_str = result_str.replace('Max. Trade Duration', '最大交易持續時間')
result_str = result_str.replace('Avg. Trade Duration', '平均交易持續時間')
print(result_str)
bt.plot()

while True:
    choice = input("按下 'Q' 退出或 'C' 繼續查詢：").strip().upper()
    if choice == 'Q':
        break
    elif choice == 'C':
        print("請提供新的參數以便繼續查詢...")
    else:
        print("無效選擇，請重新選擇。")