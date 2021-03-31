#+ 因子投资研究
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from utils import *
import statsmodels.api as sm
from scipy.linalg import norm
from scipy import stats
from numpy.random import multinomial, normal
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ANN_COEF = {'d': 252, 'w': 52, 'm': 12, 'h': 2, 'y': 1}
set_matplotlib()

class FactorUtils(object):
    """
    多因子研究中的常用函数
    """
    def __init__(self):
        pass

    @staticmethod
    def read_folder(fp, wind_footprint=None, csv_encoding='gbk'):
        """
        用于读取整个文件夹中的所有csv/xlsx文件
        :param fp: str, 文件夹路径
        :param wind_footprint: int, 是否wind来源的数据 通常为倒数第二行(2)
        :param csv_encoding: str, csv文件的encoding方式
        """
        collection = []
        fns = os.listdir(fp)

        for f in fns:
            file_type = f[-3:].lower()
            if file_type == 'csv':
                tmp = pd.read_csv(fp+f, encoding=csv_encoding)
            elif file_type == 'lsx' or file_type == 'xls':
                tmp = pd.read_excel(fp+f)
            else:
                print("%s is not csv or excel file;" % f)
                continue
            if wind_footprint is not None:
                tmp = tmp.iloc[:(-wind_footprint), :]
            collection.append(tmp)
        
        return pd.concat(collection)

    @staticmethod
    def concat_lol(list_of_arr):
        """
        拼接list of array
        """
        collection = []
        for arr in list_of_arr:
            collection.extend(list(arr))
        return collection

    @staticmethod
    def resample_price_data(df, trade_calendar):
        """
        按指定频率重新采样价格数据
        :param df: DataFrame, 长型价格数据
        :param trade_calendar: list, 指定频率的交易日历
        """
        price_data = df.pivot(index='windCode', columns='tradeDate', values='factor_val')
        new_index = trade_calendar
        union_index = pd.Index(new_index).union(price_data.columns)
        price_data = price_data.T.reindex(union_index).sort_index().fillna(method='ffill', axis=0).reindex(new_index).T
        price_data = price_data.reset_index().melt(id_vars='windCode', var_name='tradeDate', value_name='factor_val').dropna()
        price_data['factor_name'] = 'price'
        return price_data

    @staticmethod
    def price2return(price_data, future=True):
        """
        将price转换为return
        :param price_data: DataFrame, 长型价格数据
        :param future: Bool, 是否作为标签(记录未来一期的return)
        """
        price_data = price_data.pivot(index='windCode', columns='tradeDate', values='factor_val')
        if future:
            ret_data = (price_data.shift(-1, axis=1) / price_data) - 1.0
        else:
            ret_data = (price_data / price_data.shift(1, axis=1)) - 1.0
        ret_data = ret_data.reset_index().melt(id_vars='windCode', var_name='tradeDate', value_name='factor_val').dropna()
        ret_data['factor_name'] = 'return'
        return ret_data
        
    @staticmethod
    def value2quantile(series):
        """
        数值转换为排名分位数
        """
        rank = series.rank()
        rank_min = min(rank)
        rank_max = max(rank)
        return (rank - rank_min) / (rank_max - rank_min)

    @staticmethod
    def get_group_res(group_df, keep, factor):
        tmp = group_df.copy()
        value = FactorUtils.value2quantile(group_df[factor])
        tmp[factor] = value
        res = tmp.set_index(keep)[factor]
        return res

    @staticmethod
    def orthogonalize(factor_long, cross_sec, factor, control_var, method='quantile', keep='代码'):
        """
        因子正交化
        :param factor_long: DataFrame, 因子值长型矩阵
        :param cross_sec: str, 表示截面期的colname
        :param factor: str, 表示因子值的colname
        :param control_var: str/list, 用于正交化的基的colname(s)
        :param method: str, quantile是针对单个control_var时，分组将因子值转换为分位数；regression是针对多个/连续型control_var
        时，OLS求取残差
        :param keep: str, 代码
        """
        if method == 'quantile':
            if isinstance(control_var, str):
                control_var = [control_var,]
            converted = factor_long.groupby([cross_sec]+control_var).apply(lambda x: FactorUtils.get_group_res(x, keep, factor)).reset_index()

        return converted

    @staticmethod
    def preprocess(factor_long, standardize=True, cap_=True, extreme=5):
        """
        单个因子在截面上的异常值处理和标准化
        :param factor_long: DataFrame, 长型因子数据(单因子单截面)，通常为groupby的元素
        :param standardize: bool, 是否要标准化
        :param cap: bool, 是否要处理异常值
        :param extreme: int, 异常值定义中离中位数的四分位距倍数
        """
        tmp = factor_long.copy()
        if cap_:
            tmp['factor_val'] = cap(tmp['factor_val'], extreme)
        if standardize:
            tmp['factor_val'] = (tmp['factor_val'] - tmp['factor_val'].mean()) / tmp['factor_val'].std()
        return tmp

    @staticmethod
    def my_pearsonr(x, y):
        """
        nan tolerant pearsonr
        """
        nas = x.isna() | y.isna()
        corr = pearsonr(x[~nas], y[~nas])
        return corr

    @staticmethod
    def sharpe(returns, freq, rf=0.03):
        adj_factor = ANN_COEF[freq]
        adj_rf = rf / adj_factor
        adj_returns = returns - adj_rf
        
        res = np.nanmean(adj_returns) * np.sqrt(adj_factor) / np.nanstd(adj_returns, ddof=1)
        return res

    @staticmethod
    def max_drawdown(returns):
        cum_nv = (returns+1).cumprod()
        peak = cum_nv.expanding(min_periods=1).max()
        dd = (cum_nv/peak)-1
        return dd.min()

    @staticmethod
    def calmar(returns, freq):
        adj_factor = ANN_COEF[freq]
        er = np.nanmean(returns) * adj_factor
        md = FactorUtils.max_drawdown(returns)
        return er / np.abs(md)


class FactorData(object):
    """
    (多)因子数据结构
    以长型矩阵(dataframe)存储
    可生成面板、截面、时间序列回归所需的数据集generator
    TODO: 展示极端值和缺失值情况
    """
    def __init__(self, factor_long):
        """
        :param factor_long: 长型因子数据
        """
        self.factor_long = factor_long

        calendar = self.factor_long['tradeDate'].unique()
        calendar.sort()
        self.calendar = pd.Series(calendar)

    def preprocess(self, standardize=True, cap=True, extreme=5):
        """
        读取后预处理
        :param standardize: bool, 是否要标准化
        :param cap: bool, 是否要处理异常值
        :param extreme: int, 异常值定义中离中位数的四分位距倍数
        """
        self.factor_long = self.factor_long.groupby(['tradeDate', 'factor_name']).apply(lambda x: FactorUtils.preprocess(x, standardize, cap, extreme))
        
    def make_panel_data(self, X_names, y_name, start=None, end=None):
        """
        转换为面板格式
        :param X_names: list, X的col_names
        :param y_name: str, y的col_name
        :param start/end: str, 选取数据的时间段
        """
        sub = self.factor_long
        sub = sub.pivot(index=['windCode', 'tradeDate'], columns='factor_name', values='factor_val').reset_index()
        sub = sub.sort_values('tradeDate')
        sub = sub.groupby('windCode').apply(lambda x: x.fillna(method='ffill')).dropna()
        if start is not None:
            sub = sub.query("`tradeDate`>='%s'" % start)
        if end is not None:
            sub = sub.query("`tradeDate`<='%s'" % end)
        return sub[X_names], sub[y_name]

    def gen_tsreg_cut(self, cut_points):
        """
        为每个代码生成时间序列数据，以切割点切割区间
        :param cut_points: list, 时间切割点，如季末
        """
        factor_long = self.factor_long

        factor_return = factor_long.query("`windCode`=='M'")
        X_names = factor_return['factor_name'].unique().tolist()
        y_name = 'factor_val'
        asset_return = factor_long.query("`windCode`!='M'")
        factor_return = factor_return.pivot(index='tradeDate', columns='factor_name', values='factor_val').reset_index()
        all_data_wide = pd.merge(left=asset_return, right=factor_return, left_on='tradeDate', right_on='tradeDate', how='left').sort_values('tradeDate')

        for code, tmp in all_data_wide.groupby('windCode'):
            tmp = tmp.reset_index(drop=True).copy()
            time_ticker = pd.Series(0, index=tmp.index)
            change_points = tmp['tradeDate'].searchsorted(cut_points, side='right')
            time_ticker.loc[change_points] = 1
            time_ticker = time_ticker.cumsum()
            tmp['time_ticker'] = time_ticker
            for _, sub in tmp.groupby('time_ticker'):
                X = sub[X_names]
                if sub.shape[0] != 0:
                    y = sub[y_name]
                else:
                    y = pd.Series([], dtype=float)
                date_range = sub['tradeDate'].tolist()
                yield (code, date_range, X, y)
    
    @staticmethod
    def get_time_ticker(date, period='m'):
        date = str(date)
        if period == 'm':
            return date[:7]
        elif period == 'y':
            return date[:4]
        elif period == 'h':
            year = date[:4]
            d = date[5:]
            if d <= '06-30':
                half = 'h1'
            else:
                half = 'h2'
            return year+half
    
    def gen_tsreg_rolling(self, period='m', look_back=250):
        """
        为每个代码生成时间序列数据，常用于回归计算beta,alpha
        :param period: str, d/w/m/h/y, 计算频率
        :param look_back: int, 每次回归所用的样本量
        """
        factor_long = self.factor_long

        factor_return = factor_long.query("`windCode`=='M'")
        X_names = factor_return['factor_name'].unique().tolist()
        y_name = 'factor_val'
        asset_return = factor_long.query("`windCode`!='M'")
        factor_return = factor_return.pivot(index='tradeDate', columns='factor_name', values='factor_val').reset_index()
        all_data_wide = pd.merge(left=asset_return, right=factor_return, left_on='tradeDate', right_on='tradeDate', how='left').sort_values('tradeDate')

        for code, tmp in all_data_wide.groupby('windCode'):
            tmp = tmp.reset_index(drop=True).copy()
            time_ticker = tmp['tradeDate'].apply(lambda x: self.__class__.get_time_ticker(x, period))
            change_points = time_ticker != time_ticker.shift(-1)
            change_points = change_points[change_points].index
            for x in change_points:
                sub = tmp[max(x-look_back,0):x+1]
                X = sub[X_names]
                if sub.shape[0] != 0:
                    y = sub[y_name]
                else:
                    y = pd.Series([], dtype=float)
                date_range = sub['tradeDate'].tolist()
                yield (code, date_range, X, y)

    def gen_tsreg_rolling_unsupervised(self, period='m', look_back=30):
        """
        为每个代码生成时间序列数据，常用于回归计算beta,alpha
        :param period: str, d/w/m/h/y, 计算频率
        :param look_back: int, 每次回归所用的样本量
        """
        factor_return = self.factor_long
        X_names = factor_return['factor_name'].unique().tolist()
        all_data_wide = factor_return.pivot(index='tradeDate', columns='factor_name', values='factor_val').reset_index()
        time_ticker = all_data_wide['tradeDate'].apply(lambda x: self.__class__.get_time_ticker(x, period))
        change_points = time_ticker != time_ticker.shift(-1)
        change_points = change_points[change_points].index

        for x in change_points:
            sub = all_data_wide[max(x-look_back,0):x+1]
            X = sub[X_names]
            date_range = sub['tradeDate'].tolist()
            yield (date_range, X)

    def gen_cross_sec_panel(self, X_names, y_name, start=None, end=None):
        """
        用于生成截面单因子回归测试所用panel
        :param X_names: list
        :param y_name: str
        :param start: str
        :param end: str
        """
        sub = self.factor_long
        sub = sub.pivot(index=['windCode', 'tradeDate'], columns='factor_name', values='factor_val').reset_index()
        sub = sub.sort_values('tradeDate')
        sub = sub.groupby('windCode').apply(lambda x: x.fillna(method='ffill')).dropna()
        if start is not None:
            sub = sub.query("`tradeDate`>='%s'" % start)
        if end is not None:
            sub = sub.query("`tradeDate`<='%s'" % end)
        cross_secs = sub.groupby('tradeDate')
        for date, full in cross_secs:
            yield(date, full[X_names], full[y_name])


class FactorTest(object):
    """
    单因子分层回测和IC检测
    前者可以看出非线性关系，而IC检测不行；
    当分层回测显示出非线性关系时，应该调整因子encoding方式，或者放弃IC直接做piecewise回归检验
    """
    def __init__(self, api, factor_dict, return_dict, freq='m', start=None, end=None, standardize=True, cap=True, extreme=5):
        """
        初始化
        :param api: 数据库接口
        :param factor_dict: dict, {表名: [因子名,]}
        :param return_dict: dict, {表名: [价格名,]}
        :param freq: str, 测试频率
        :param start/end: date, 读取数据的起始/结束时间 
        :param standardize: bool, 是否要标准化
        :param cap: bool, 是否要处理异常值
        :param extreme: int, 异常值定义中离中位数的四分位距倍数
        """
        self.api = api
        self.standardize = standardize
        self.cap = cap
        self.extreme = extreme
        self.update_return_data(return_dict, freq, start, end)
        self.update_factor_data(factor_dict)
        self.freq = freq

        # 默认测试区间
        self.start = self.calendar[0]
        self.end = self.calendar[-1]

    def update_factor_data(self, factor_dict):
        """
        读取/重新读取因子数据
        :param factor_dict: dict, {表名: [因子名,]}
        """
        api = self.api
        self.factor_data = api.get_factors(factor_dict)
        self.factor_data.groupby(['tradeDate', 'factor_name']).apply(lambda x: FactorUtils.preprocess(x, self.standardize, self.cap, self.extreme))
        self.factor_df = self.factor_data.pivot(index='windCode', columns='tradeDate', values='factor_val')
        self.fac_name = list(factor_dict.values())[0][0]

        # 对齐因子和价格数据
        union_index = self.return_data.columns.union(self.factor_df.columns).sort_values()
        self.factor_df = self.factor_df.T.reindex(union_index, method='ffill').reindex(self.return_data.columns).T
        self.calendar = self.return_data.columns

    def update_return_data(self, return_dict, freq, start, end):
        """
        读取/重新读取收益率数据
        :param return_dict: dict, {表名: [价格名,]}
        """
        api = self.api
        self.return_data = api.get_price_quote(return_dict, freq, start=start, end=end).pivot(index='windCode', columns='tradeDate', values='factor_val')
        
    def _portfolio_return_single_period(self, codes, period, weight=None):
        """
        计算单个截面期的组合收益率，默认等权组合
        """
        selected = self.return_data.loc[codes][period]
        if len(selected) == 0:
            return 0.0
        if weight is None:
            pr = np.mean(selected)
        else:
            pr = np.sum(selected.values * weight)
        return pr
    
    def _get_group_codes(self, period, group_num=3):
        """
        对某个截面按因子值大小分层
        """
        selected = self.factor_df[period].dropna().sort_values()
        groups = np.array_split(selected.index, group_num)
        return groups

    def _get_group_codes_level(self, period, unique_vals):
        """
        对于离散型因子，按所有level分层
        """
        selected = self.factor_df[period].dropna()
        groups = []
        for val in unique_vals:
            groups.append(selected[selected==val].index)
        return groups
        
    def calc_group_cum_return(self, start=None, end=None, group_num=5, compound=True, ar=True, summary=False, discrete=False):
        """
        对区间中的每个截面分层，计算每层累计收益
        :param start/end: date, 回测起点/终点
        :param group_num: int, 分层数
        :param compound: bool, 是否以复利计算，单利计算可以避免初始阶段区别造成的巨大影响
        :param ar: bool，是否以当期所有资产等权组合的收益作为基准计算超额收益
        :param summary: bool，是否返回各层组合的sharpe ratio, max_drawdown, calmar ratio
        :param discrete: bool, 是否是离散型变量，需要改变分层方法
        TODO: 多空组合的收益曲线、sharpe比例；但在基金中用不到，因为难以做空
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        port_returns = []
        base_returns = []
        c = self.calendar
        periods = c[c.get_loc(start):c.get_loc(end)]
        ed = c[c.get_loc(end)]

        if discrete:
            all_vals = self.factor_df.values.flatten()
            all_vals = all_vals[~np.isnan(all_vals)]
            unique_vals = np.sort(pd.unique(all_vals))
        
        for period in periods:
            tmp = []
            if discrete:
                period_groups = self._get_group_codes_level(period, unique_vals)
            else:
                period_groups = self._get_group_codes(period, group_num=group_num)
            period_all_codes = FactorUtils.concat_lol(period_groups)
            br = self._portfolio_return_single_period(period_all_codes, period)
            base_returns.append(br)
            for group in period_groups:
                cross_return = self._portfolio_return_single_period(group, period)
                tmp.append(cross_return)
            port_returns.append(tmp)
        
        port_returns = pd.DataFrame(port_returns, index=periods)
        base_returns = pd.Series(base_returns, index=periods)
        port_returns.loc[ed] = [np.nan] * group_num
        base_returns.loc[ed] = np.nan

        if compound:
            crt = (port_returns.shift(1).fillna(0.0) + 1.0).cumprod()
            brt = (base_returns.shift(1).fillna(0.0) + 1.0).cumprod()
        else:
            crt = port_returns.shift(1).fillna(0.0).cumsum()
            brt = base_returns.shift(1).fillna(0.0).cumsum()
        
        if summary:
            freq = self.freq
            pft = crt.iloc[-1, :] / crt.iloc[0, :] - 1.0
            sr = port_returns.apply(lambda x: FactorUtils.sharpe(x, freq))
            md = port_returns.apply(lambda x: FactorUtils.max_drawdown(x))
            cr = port_returns.apply(lambda x: FactorUtils.calmar(x, freq))
            summary_stat = pd.concat([pft, sr, md, cr], axis=1)
            summary_stat.columns = ['收益率', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio']
            return summary_stat

        if ar:
            crt = crt.apply(lambda x: x - brt)
        return crt
    
    def plot_group_cum_return(self, start=None, end=None, group_num=5, compound=True, ar=True, discrete=False):
        """
        绘制分层回测图
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        
        fig, ax = whiteboard()
        crt = self.calc_group_cum_return(start, end, group_num, compound, ar, discrete)
        crt.plot(ax=ax, linewidth=1.1)
        ax.set_ylabel('累计值')
        ax.set_title('%s分层回测' % self.fac_name)

    def calc_IC(self, period, rank=True):
        """
        计算IC值
        :param period: date, 日期
        :param rank: bool, 是否计算rank IC
        """
        if rank:
            corr = spearmanr
        else:
            corr = FactorUtils.my_pearsonr
        fval = self.factor_df[period].dropna()
        rt = self.return_data[period].reindex(fval.index)
        return corr(fval, rt)[0]

    def calc_IC_series(self, start=None, end=None, rank=True, summary=False):
        """
        计算IC序列
        :param start/end: date, 计算IC序列的时间区间
        :param rank: bool, 是否计算rank IC
        :param summary: bool, 是否返回IC序列的统计量
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        IC_series = []
        c = self.calendar
        periods = c[c.get_loc(start):c.get_loc(end)+1]
        for period in periods:
            IC_series.append(self.calc_IC(period, rank=rank))
        IC_series = pd.Series(IC_series, index=periods)

        if summary:
            summary_stats = {}
            summary_stats['IC_mean'] = IC_series.mean()
            summary_stats['IC_IR'] = IC_series.mean() / IC_series.std()
            summary_stats['IC_sig_ratio'] = (IC_series > 0).sum() / len(IC_series)
            return summary_stats
        else:
            return IC_series

    def plot_IC_series(self, start=None, end=None, rank=True):
        """
        绘制IC序列图
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if rank:
            prefix = 'Rank '
        else:
            prefix = ''
            
        IC_ser = self.calc_IC_series(start, end)
        IC_ser.index = [str(x) for x in IC_ser.index]
        fig, ax = whiteboard()
        ax2 = ax.twinx()
        plot_ = IC_ser.plot.bar(ax=ax, label=prefix+'IC')
        plot_2 = IC_ser.cumsum().plot(ax=ax2, linewidth=1.1, color='orange', label=prefix+'IC累计值')
        sns_sparse_xticks(plot_, 10)
        fig.autofmt_xdate()
        ax.legend()
        ax2.legend()

        ax.set_xlabel('日期')
        ax.set_ylabel(prefix+'IC')
        ax.set_title('%s IC图' % self.fac_name)


class FactorReg(object):
    """
    多因子线性回归检验框架
    """
    def __init__(self, api, factor_dict, return_dict, freq='m', start=None, end=None):
        """
        初始化
        :param api: 数据库接口
        :param factor_dict: dict, {表名: [因子名1, 因子名2, ...]}
        :param return_dict: dict, {表名: [价格名,]}
        :param freq: str, 测试频率
        :param start/end: date, 读取数据的起始/结束时间 
        """
        self.api = api
        return_data = api.get_price_quote(return_dict, freq, start=start, end=end)
        factor_data = api.get_factors(factor_dict)
        factor_long = pd.concat([factor_data, return_data])
        self.factor_data = FactorData(factor_long)
        self.calendar = self.factor_data.calendar
    
    def cross_sec_reg_test(self, X_names, y_name='return', start=None, end=None, summary=False):
        """
        对每个截面做回归，并汇总
        :param X_names: list, 因子名称
        :param y_name: str, target名称(通常是下期收益率)
        :param start/end: date, 回归测试的时间区间
        :param summary: 是否返回t值序列的统计量及Fama_Macbeth检验统计量
        """
        fd = self.factor_data
        cross_secs = fd.gen_cross_sec_panel(X_names, y_name, start=start, end=end)
        collection = []
        col_names = ['tradeDate', 'const'] + X_names + ['t_const'] + ['t_'+x for x in X_names] + ['F_val']
        for date, X, y in cross_secs:
            tmp = [date,]
            X = sm.add_constant(X)
            model = sm.OLS(y,X)
            results = model.fit(cov_type='HC3')  # 控制异方差
            tmp.extend(results.params.tolist())
            tmp.extend(results.tvalues.tolist())
            tmp.append(results.fvalue)
            collection.append(tmp)
        result = pd.DataFrame(collection, columns=col_names)
        self.reg_result = result

        if summary:
            t_X_names = ['t_'+x for x in X_names]
            summary_stats = {}
            summary_stats['t_abs_mean'] = result[t_X_names].abs().mean()
            summary_stats['t_sig_ratio'] = (result[t_X_names].abs() > 2).sum() / result.shape[0]
            summary_stats['t_mean'] = result[t_X_names].mean()
            summary_stats['fama_macbeth_test'] = result[X_names].mean() / result[X_names].std()
            return summary_stats
        else:
            return result

    def plot_t_bars(self, factor_name):
        """
        绘制单个因子的t值序列图
        :param factor_name: str, 因子名
        """
        fig, ax = whiteboard()
        plot_ = self.reg_result.set_index('tradeDate')['t_%s' % factor_name].plot.bar(ax=ax)
        sns_sparse_xticks(plot_, 10)
        fig.autofmt_xdate()
        ax.axhline(y=2,linestyle ='--', color='red', linewidth=1)
        ax.axhline(y=-2,linestyle ='--', color='red', linewidth=1)
        ax.set_ylabel('t-value')
        ax.set_title('%s因子截面回归t值图'%factor_name)

class FactorBlend(object):
    """
    因子合成框架
    1. 线性合成 - max_IR方法
    2. 机器学习合成 - 结合Optuna自动选择参数 (TODO)
    """
    def __init__(self, api, factor_dict_all, return_dict, start=None, end=None):
        """
        :param api: 数据库接口
        :param factor_dict_all: dict, 用于合成的底层因子字典
        :param return_dict: dict, 收益率字典
        :param start/end: date, 读取数据的时间区间
        """
        self.api = api
        self.names = FactorUtils.concat_lol(list(factor_dict_all.values()))
        self.factor_dict_all = factor_dict_all
        self.return_dict = return_dict
        self.start = start
        self.end = end
        
    def load_data(self, rank=False):
        """
        读取数据，合并，计算ic
        :param rank: bool, 是否用rank IC，理论上错误但实际上可能有用
        """
        factor_dict_all, return_dict = self.factor_dict_all, self.return_dict
        api, start, end = self.api, self.start, self.end
        names = self.names
        factor_data_all = []
        ic_ser_all = []
        factor_num = 0

        for table_name, factor_list in factor_dict_all.items():
            for factor_name in factor_list:
                factor_dict = {table_name: [factor_name,]}
                if factor_num == 0:
                    factor_test = FactorTest(api, factor_dict, return_dict)
                else:
                    factor_test.update_factor_data(factor_dict)
                ic_ser = factor_test.calc_IC_series(start, end, rank=rank)
                ic_ser_all.append(ic_ser)
                factor_data_all.append(factor_test.factor_data)
                factor_num += 1

        ic_ser_all = pd.concat(ic_ser_all, axis=1)
        ic_ser_all.columns = names
        ic_ser_all = ic_ser_all.reset_index().melt(id_vars='tradeDate', var_name='factor_name', value_name='factor_val')
        ic_ser_all['windCode'] = 'M'
        
        self.factor_data_all = pd.concat(factor_data_all)
        self.ic_ser_all = FactorData(ic_ser_all)
        self.factor_num = factor_num
        
        self.AVG_WEIGHT = 1.0 / factor_num
        self.return_data = factor_test.return_data
        
    def blend_ic_ir(self, blended_name, period='m', look_back=30, ewa=False, rank=False, positive=False):
        """
        :param blended_name: str, 合成因子的名称
        :param period: str, 合成周期
        :param look_back: int, 用于估计IR的时序样本量
        :param ewa: bool, 估计IC均值时是否用exponential weight
        :param rank: bool, 是否使用rank IC
        :param positive: bool, 是否使用权重非负约束
        """
        print('loading data...')
        self.load_data(rank)
        print('done;')
        
        ic_ser_all = self.ic_ser_all
        names = self.names
        AVG_WEIGHT = self.AVG_WEIGHT
        factor_num = self.factor_num
        factor_data_all = self.factor_data_all
        
        ic_ts_generator = ic_ser_all.gen_tsreg_rolling_unsupervised(period, look_back)
        weight_collection = []
        date_collection = []
        
        print('calculating blend weights...')
        for dr, X in ic_ts_generator:
            X = X.fillna(0.0)
            cov = LedoitWolf().fit(X).covariance_
            if ewa:
                ic_mean = X.ewm(halflife=0.25*len(X)).mean().values[-1]
            else:
                ic_mean = X.mean().values
            fun = lambda w: - w.T.dot(ic_mean) / np.sqrt(w.T.dot(cov).dot(w))
            init = np.array([AVG_WEIGHT, ] * factor_num)
            cons = ({'type': 'ineq', 'fun': lambda x: x})
            if positive:
                res = minimize(fun, init, method='SLSQP', constraints=cons).x
            else:
                res = minimize(fun, init, method='SLSQP').x
            res = res / np.sqrt(np.sum(res ** 2))
            weight_collection.append(res.tolist())
            date_collection.append(dr[-1])
        print('done;')
        
        weight_collection = pd.DataFrame(weight_collection, index=date_collection, columns=names)
        weight_collection = weight_collection.fillna(AVG_WEIGHT)
        self.weight_collection = weight_collection.copy()
        weight_collection = weight_collection.reset_index().melt(id_vars='index', var_name='factor_name', value_name='weight').rename(columns={'index': 'tradeDate'})
        merged = pd.merge(left=factor_data_all, right=weight_collection, left_on=['tradeDate', 'factor_name'], right_on=['tradeDate', 'factor_name'], how='left')
        merged['factor_val'] = merged['factor_val'] * merged['weight']

        blended_factor_long = merged.dropna().groupby(['tradeDate', 'windCode'])['factor_val'].sum().reset_index()
        blended_factor_long['factor_name'] = blended_name
        self.blended_factor_long = blended_factor_long
        print('blended factor %s ready;' % blended_name)

    def plot_weight(self):
        """
        绘制合成权重的序列图
        """
        fig, ax = whiteboard()
        self.weight_collection.plot(ax=ax, linewidth=1.1)
        ax.set_xlabel('日期')
        ax.set_ylabel('因子权重')
        ax.set_title('合成权重序列')

    def save(self, table_name):
        """
        保存因子
        :param table_name: str, 保存的表名
        """
        api = self.api
        api.save_factor_data(self.blended_factor_long, table_name)