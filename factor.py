#+ 因子投资研究
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from .utils import *
import statsmodels.api as sm
from scipy.linalg import norm
from scipy import stats
from numpy.random import multinomial, normal

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
    def winsorize():
        pass


class FactorData(object):
    """
    (多)因子数据结构
    以长型矩阵(dataframe)存储
    可生成面板、截面、时间序列回归所需的数据集generator
    """
    def __init__(self, factor_long):
        """
        :param factor_long: 长型因子数据
        """
        self.factor_long = factor_long

        calendar = self.factor_long['tradeDate'].unique()
        calendar.sort()
        self.calendar = pd.Series(calendar)

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

        factor_return = factor_long.query("`code`=='M'")
        X_names = factor_return['factor_name'].unique().tolist()
        y_name = 'factor_val'
        asset_return = factor_long.query("`code`!='M'")
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

        factor_return = factor_long.query("`code`=='M'")
        X_names = factor_return['factor_name'].unique().tolist()
        y_name = 'factor_val'
        asset_return = factor_long.query("`code`!='M'")
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
    def __init__(self, api, factor_dict, return_dict, freq='m', start=None, end=None):
        """
        初始化
        :param api: 数据库接口
        :param factor_dict: dict, {表名: [因子名,]}
        :param return_dict: dict, {表名: [价格名,]}
        :param freq: str, 测试频率
        :param start/end: date, 读取数据的起始/结束时间 
        """
        self.api = api
        self.return_data = api.get_price_quote(return_dict, freq, start=start, end=end).pivot(index='windCode', columns='tradeDate', values='factor_val')
        factor_data = api.get_factors(factor_dict)
        self.factor_df = factor_data.pivot(index='windCode', columns='tradeDate', values='factor_val')
        self.fac_name = list(factor_dict.values())[0][0]
        
        # 对齐因子和价格数据
        union_index = self.return_data.columns.union(self.factor_df.columns).sort_values()
        self.factor_df = self.factor_df.T.reindex(union_index, method='ffill').reindex(self.return_data.columns).T
        self.calendar = self.return_data.columns
        
    def _portfolio_return_single_period(self, codes, period, weight=None):
        """
        计算单个截面期的组合收益率，默认等权组合
        """
        selected = self.return_data.loc[codes][period]
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
        
    def calc_group_cum_return(self, start, end, group_num=5, compound=True, ar=True):
        """
        对区间中的每个截面分层，计算每层累计收益
        :param start/end: date, 回测起点/终点
        :param group_num: int, 分层数
        :param compound: bool, 是否以复利计算，单利计算可以避免初始阶段区别造成的巨大影响
        :param ar: bool，是否以当期所有资产等权组合的收益作为基准计算超额收益
        """
        port_returns = []
        base_returns = []
        c = self.calendar
        periods = c[c.get_loc(start):c.get_loc(end)+1]
        ed = c[c.get_loc(end)+1]
        
        for period in periods:
            tmp = []
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

        if ar:
            crt = crt.apply(lambda x: x - brt)
        return crt
    
    def plot_group_cum_return(self, start, end, group_num=3, compound=True, ar=True):
        """
        绘制分层回测图
        """
        fig, ax = whiteboard()
        crt = self.calc_group_cum_return(start, end, group_num, compound, ar)
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
            corr = pearsonr
        fval = self.factor_df[period].dropna()
        rt = self.return_data[period].reindex(fval.index)
        return corr(fval, rt)[0]

    def calc_IC_series(self, start, end, rank=True, summary=False):
        """
        计算IC序列
        :param start/end: date, 计算IC序列的时间区间
        :param rank: bool, 是否计算rank IC
        :param summary: bool, 是否返回IC序列的统计量
        """
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

    def IC_series_plot(self, start, end, rank=True):
        """
        绘制IC序列图
        """
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
    
    def cross_sec_reg_test(self, X_names, y_name, start=None, end=None, summary=False):
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