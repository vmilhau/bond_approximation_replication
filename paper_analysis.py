import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pandas._libs.tslibs.parsing import DateParseError
from typing import Dict, Optional, Sequence, Callable
from bond import CouponBond, ZeroCouponBond
from pathlib import Path
from collections.abc import Callable, Mapping


# ===== CUSTOM EXCEPTIONS =====

class MissingMaturityDateError(Exception):
    """Exception raised when maturity date is missing."""
    pass


class NegativeCouponRateError(Exception):
    """Exception raised when coupon rate is negative."""
    pass


class ZeroCouponRateError(Exception):
    """Exception raised when coupon rate is zero for a non-zero-coupon bond."""
    pass


class IssueDateAfterMaturityDateError(Exception):
    """Exception raised when issue date is after maturity date."""
    pass


# ===== IO FUNCTIONS =====

def read_bond_dsc(file_path: Path) -> pd.DataFrame:
    """
    Read bond description data from a pipe-delimited file and return it as a DataFrame.

    Parameters:
    file_path (str): The path to the pipe-delimited file containing bond data.

    Returns:
    A DataFrame containing the bond data.
    """
    bond_data = pd.read_csv(file_path, 
                            sep='|',
                            parse_dates=[1, 2],
                            names=['Identifier', 'Issue Date', 'Maturity Date', 
                                    'Rate 1', 'Rate 2', 'Coupon Frequency',
                                    'Amount 1', 'Amount 2', 'Rating 1',
                                    'Indicator', 'Country 1', 'Country 2']
                            )

    return bond_data


def read_bond_prc(file_path: Path) -> pd.DataFrame:
    """
    Read bond price data from a pipe-delimited file and return it as a DataFrame.

    Parameters:
    file_path (str): The path to the pipe-delimited file containing bond price data.

    Returns:
    A DataFrame containing the bond price data.
    """
    bond_data = pd.read_csv(file_path, 
                            sep='|',
                            parse_dates=[1],
                            names=['Identifier', 'Pricing Date', 'Bid', 'Ask']
                            )

    return bond_data


def read_bond_typ(file_path: Path) -> pd.DataFrame:
    """
    Read bond type data from a comma-delimited file and return it as a DataFrame.

    Parameters:
    file_path (str): The path to the comma-delimited file containing bond type data.

    Returns:
    A DataFrame containing the bond type data.
    """
    bond_data = pd.read_csv(file_path, 
                            sep='|',
                            names=['Identifier', 'Type']
                            )

    return bond_data


def read_bond_auction_data(file_path: Path) -> pd.DataFrame:
    """
    Read bond auction data from Fed FiscalData.
    """
    headers = pd.read_csv(file_path,
                         sep=',',
                         nrows=1)
    
    date_headers = [x for x in headers if isinstance(x, str) and x.endswith('_date')]

    bond_data = pd.read_csv(file_path, 
                            sep=',',
                            parse_dates=date_headers,
                            index_col=0
                            )

    return bond_data


def read_rate_data(file_path: Path) -> pd.DataFrame:
    """
    Read data on constant-maturity rates from FRED.
    """
    rat_data = pd.read_csv(file_path, 
                           sep=',',
                           parse_dates=[0],
                           index_col=0
                           )
    
    return rat_data


# ===== BOND CREATION =====

def create_bond_from_raw(dsc: pd.DataFrame, auc: pd.DataFrame) -> Dict[str, ZeroCouponBond | CouponBond]:
    """
    Create a dictionary of Bond instances from a DataFrame of bond description data.

    Parameters:
    bond_dsc_data (pd.DataFrame): A DataFrame containing bond description data.

    Returns:
    A dictionary of Bond instances.
    """
    cusips = dsc.index.unique()

    def create_bond(row_dsc: pd.Series, rows_auc: pd.DataFrame) -> Optional[ZeroCouponBond | CouponBond]:
        try:
            maturity_date = pd.Timestamp(row_dsc['Maturity Date'])

            if pd.isnull(maturity_date):
                raise(MissingMaturityDateError)
            
            issue_date_dsc = pd.Timestamp(row_dsc['Issue Date'])
            issue_dates_auc = pd.DatetimeIndex(rows_auc['issue_date'])

            # We keep the issue dates from the bond auction file if at least one date is provided
            # Else we keep the issue date from the bond description file
            if all(issue_dates_auc.isna()):
                issue_dates = pd.DatetimeIndex([issue_date_dsc])
            else:
                issue_dates = pd.DatetimeIndex(issue_dates_auc.sort_values())
            
            if issue_dates[-1] >= maturity_date:
                raise(IssueDateAfterMaturityDateError)

            coupon_rate = float(row_dsc['Rate 1'])

            if coupon_rate < 0.0:
                raise(NegativeCouponRateError)
            
            coupon_frequency = row_dsc['Coupon Frequency']

            if coupon_frequency == 'B':

                # The bond is a zero-coupon bond
                return ZeroCouponBond(
                    face_value=100.0,
                    maturity_date=maturity_date,
                    issue_dates=issue_dates,
                    cusip=str(row_dsc.name)
                )
            
            else:
                coupon_rate_pct = float(row_dsc['Rate 1'])

                # If the coupon rate is 0 but the bond is not flagged as a zero coupon, it may be a floating-rate note
                if coupon_rate_pct == 0.0:
                    raise(ZeroCouponRateError)

                return CouponBond(
                    face_value=100.0,
                    maturity_date=maturity_date,
                    coupon_rate_pct=coupon_rate,
                    coupon_frequency=row_dsc['Coupon Frequency'],
                    issue_dates=issue_dates,
                    cusip=str(row_dsc.name)
                )

        except (DateParseError, ValueError, 
                MissingMaturityDateError, NegativeCouponRateError, 
                IssueDateAfterMaturityDateError, ZeroCouponRateError):
            return None
        

    d = {}
    
    for cusip in cusips:
        b = create_bond(dsc.loc[cusip], auc.loc[[cusip]])

        if b:
            d[cusip] = b              

    return d


# ===== DATA CLEANING UTILITIES =====

def drop_invalid_index(b: ZeroCouponBond | CouponBond, data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are outside the bond's issue and maturity dates.
    """
    to_drop = (data.index < b.issue_dates[0]) | (data.index > b.maturity_date)
    return data.drop(data.index[to_drop], axis=0)


def fill_business_days(prc: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing business days in the price DataFrame by forward-filling the last available price.

    Parameters:
    prc (pd.DataFrame): A DataFrame containing bond price data indexed by date.
    Index is assumed to be sorted.

    Returns:
    A DataFrame with missing business days filled.
    """
    if len(prc) >= 1:
        # The DataFrame has at least one row
        all_business_days = (pd.date_range(start=prc.index[0], end=prc.index[-1], freq='B')
                            .rename(prc.index.name)  # Keep original index name if present
                            )
        
        prc_filled = prc.reindex(all_business_days).ffill()
    else:
        # Return the zero-row DataFrame
        prc_filled = prc

    return prc_filled


def aggregate_bid_ask(bid: pd.Series, ask: pd.Series) -> pd.Series:
    one_zero = (bid == 0.0) | (ask == 0.0)
    two_positive = (bid > 0.0) & (ask > 0.0)
    price = np.zeros_like(bid)
    price[one_zero] = np.maximum(bid[one_zero], ask[one_zero])
    price[two_positive] = 0.5 * (bid[two_positive] + ask[two_positive])
    return pd.Series(price, index=bid.index)


# ===== HELPER FUNCTIONS =====

def _year_number(x: pd.Series) -> float:
    """
    Calculate the number of years from first to last index.
    
    Parameters:
    x (pd.Series): A Series with a DatetimeIndex.
    
    Returns:
    The number of years as a float.
    """
    return (x.index[-1] - x.index[0]).days / 365.25


def pick_from_first_to_last(x: list, step: int) -> list:
    """
    Pick one element out of every 'step' elements from a list, starting from the first element, and
    systematically appending the last, even if the length of the list is not a multiple of the step plus 1.

    Parameters:
    x (list): A list.
    step (int): The step size for picking items.

    Returns:
    A list.
    """
    if (len(x) - 1) % step == 0:
        return x[::step]
    else:
        y = x[::step].copy()
        y.append(x[-1])
        return y


# ===== CORE CLASSES =====

class RawBondData:
    def __init__(self, bond_dsc_data: pd.DataFrame, bond_prc_data: pd.DataFrame, bond_typ_data: pd.DataFrame, bond_auc_data: pd.DataFrame):
        self.bond_dsc_data = bond_dsc_data
        self.bond_prc_data = bond_prc_data
        self.bond_typ_data = bond_typ_data
        self.bond_auc_data = bond_auc_data


class BondData:
    def __init__(self, bond_description: Dict[str, ZeroCouponBond | CouponBond], clean_prices: pd.Series):
        """
        Initialize BondData with bond descriptions and clean prices.

        It is assumed that bond_description contains only valid Bond instances and that clean prices identifiers
        match those in bond_description.
        """
        self.bond_description = bond_description
        self.clean_prices = clean_prices

        self._holding_period = (clean_prices.groupby('Identifier')
                                            .aggregate(lambda x: _year_number(x.droplevel('Identifier')))
                                            .rename('Holding Period (years)'))


    def description(self) -> pd.Series:
        """
        Generate descriptive statistics for the bond data.

        Returns:
        A DataFrame containing descriptive statistics.
        """
        prc_index = self.clean_prices.index.get_level_values(level='Pricing Date')

        coupon_freq = pd.Series({k: v.coupon_frequency for k, v in self.bond_description.items() 
                                if isinstance(v, CouponBond)})

        first_pricing_date = prc_index.min()
        last_pricing_date = prc_index.max()

        zero_coupons = {k: v for k, v in self.bond_description.items()
                        if isinstance(v, ZeroCouponBond)}

        desc = pd.Series({'First pricing date': first_pricing_date, 
                                'Last pricing date': last_pricing_date,
                                'Bonds analyzed': len(self.bond_description),
                                'Day-price pairs': len(self.clean_prices),
                                'Zero-coupon bonds': len(zero_coupons),
                                'Maximum zero-coupon maturity': np.array([v.maturity_date - v.issue_dates[0] 
                                                                          for _, v in zero_coupons.items()]).max(),
                                'Minimum holding period (days)': self._holding_period.min() * 365.25,
                                'Maximum holding period (years)': self._holding_period.max(),
                                'Bonds with zero holding period': (self._holding_period == 0.).sum(),
                                'Minimum holding period excluding 0 (days)': self._holding_period[self._holding_period > 0.].min() * 365.25
                                })

        return pd.concat([coupon_freq.value_counts(), desc])


class CalculatedBondData:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    

    def resample(self, n: int) -> 'CalculatedBondData':
        """
        Create a resampled CalculatedBondData object by picking one row out of n within each bond sample 
        from the original object.
        """
        def pick_every_n(x: pd.DataFrame) -> pd.DataFrame:
            return x.loc[pick_from_first_to_last(x.index.tolist(), n)]

        resampled_data = (self.data.groupby('Identifier')  # Group by bond identifier
                            .apply(lambda x: pick_every_n(x.droplevel('Identifier')))
                            )

        return CalculatedBondData(resampled_data)
    

    def bond_return_decomposition(self) -> pd.DataFrame:
        """
        Decompose bond returns into contributions of income, duration and convexity.

        Parameters:
        bond_data (pd.DataFrame): A DataFrame containing columns 
        'Yield to Maturity', 'Modified Duration', and 'Modified Convexity'.

        Returns:
        A DataFrame containing the return analysis.
        """

        def decomposition_per_bond(x: pd.DataFrame) -> pd.DataFrame:
            time_step = x.index.diff().total_seconds() / (365.25 * 24 * 3600) # pyright: ignore[reportAttributeAccessIssue]

            ytm = x['Yield to Maturity'].to_numpy()
            modified_duration = x['Modified Duration'].to_numpy()
            modified_convexity = x['Modified Convexity'].to_numpy()

            income_contribution = np.empty_like(ytm)
            duration_contribution = np.empty_like(modified_duration)
            convexity_contribution = np.empty_like(modified_convexity)

            log_price_time_derivative = x['Log Price Time Derivative']
            delta_yield = np.diff(ytm)
            
            income_contribution[1:] = log_price_time_derivative[:-1] * time_step[1:]
            duration_contribution[1:] = -modified_duration[:-1] * delta_yield
            convexity_contribution[1:] = 0.5 * modified_convexity[:-1] * (delta_yield ** 2)

            income_contribution[0] = np.nan
            duration_contribution[0] = np.nan
            convexity_contribution[0] = np.nan

            return pd.DataFrame({'Income Contribution': income_contribution,
                                'Duration Contribution': duration_contribution,
                                'Convexity Contribution': convexity_contribution},
                                index=x.index)

        contributions = (self.data.groupby('Identifier')
                         .apply(lambda z: decomposition_per_bond(z.droplevel('Identifier')))
                    )
        
        return contributions
    

class RateData:
    def __init__(self, data: pd.Series):
        self.data = data


    def drop_na(self):
        return RateData(self.data.dropna())
        

    def rolling_volatility(self, window: int, step: int) -> 'RateData':
        """
        Rolling volatility of changes in interest rate.
        """
        return RateData(self.data.diff().rolling(window=window, step=step).std())


    def sub_period(self, first: pd.Timestamp, last: pd.Timestamp) -> 'RateData':
        """
        Select all datapoints with a date between first and last (endpoints included).
        """

        sub_data = self.data.loc[(self.data.index >= first) & (self.data.index <= last)]
        return RateData(sub_data)


class PaperRegimeData:
    def __init__(self, level_data: RateData, spread_data: RateData, volatility_data: RateData) -> None:
        self.level_data = level_data
        self.spread_data = spread_data
        self.volatility_data = volatility_data


class PaperRawData:
    def __init__(self, raw_bond_data: RawBondData, raw_rat_data: pd.DataFrame, raw_spr_data: pd.DataFrame) -> None:
        self.raw_bond_data = raw_bond_data
        self.raw_rat_data = raw_rat_data
        self.raw_spr_data = raw_spr_data


    def paper_rate_data(self):
        bond_dates = self.raw_bond_data.bond_prc_data.index.get_level_values('Pricing Date')

        first = bond_dates.min()
        last = bond_dates.max()

        long_level_data = RateData(self.raw_rat_data['1 Year'] / 100).drop_na()
            
        level_data = long_level_data.sub_period(first, last)
        spread_data = RateData(self.raw_spr_data['Spread'] / 100).sub_period(first, last)
        volatility_data = long_level_data.rolling_volatility(window=30, step=1).sub_period(first, last)

        return PaperRegimeData(level_data, spread_data, volatility_data)
    

# Step 1: Filter and clean raw data
def filter_and_clean(x: RawBondData) -> BondData:
    bond_dsc_data = x.bond_dsc_data.set_index('Identifier')
    bond_auc_data = x.bond_auc_data.set_index('cusip')

    admissibility_info = pd.concat((x.bond_typ_data, 
                                    bond_dsc_data['Indicator']),
                                    axis=1)
    
    # To be admissible, a bond must be flagged as a straight bond or a zero coupon (thereby excluding floating-rate notes and inflation-protected securities).
    # If no type info is given, the bond is presumed to be admissible.
    # Similarly, a bond must be marked as callable or putable.
    is_admissible = (((admissibility_info['Type'].isin(['Straight bonds',
                                                        'Zero-coupon'
                                                        ])) | 
                        pd.isna(admissibility_info['Type'])) &  # Presume that bond is admissible if no type info is given
                        ~admissibility_info['Indicator'].isin(['C', 'P', 'CP'])  # Must not be callable or putable
                        )
    
    admissible_dsc = bond_dsc_data[is_admissible[bond_dsc_data.index]]

    admissible_auc = admissible_dsc.join(bond_auc_data).loc[admissible_dsc.index]

    dsc = create_bond_from_raw(admissible_dsc,
                               admissible_auc)

    cusip_from_prc = x.bond_prc_data.index.unique(level='Identifier')
    cusip_from_dsc = dsc.keys()

    valid_cusips = cusip_from_prc.intersection(cusip_from_dsc)

    prc_list = [fill_business_days(drop_invalid_index(dsc[cusip],
                                                      x.bond_prc_data.xs(cusip, level='Identifier')))
                for cusip in valid_cusips]
    
    # Drop the empty price series
    non_empty_series = [(cusip, prc) for cusip, prc in zip(valid_cusips, prc_list) if len(prc) >= 1]
    
    final_cusips = [x[0] for x in non_empty_series]
    final_prc_list = [x[1] for x in non_empty_series]

    prc = pd.concat(final_prc_list, axis=0, keys=final_cusips, names=['Identifier', 'Pricing Date'])

    clean_prices = aggregate_bid_ask(prc['Bid'], prc['Ask']).rename('Clean Price')

    return BondData(bond_description={k: dsc[k] for k in final_cusips}, # pyright: ignore[reportArgumentType]
                    clean_prices=clean_prices)

    
# Step 2: Calculate dirty price, total return index, YTM, duration and convexity
def bond_calculation(x: BondData) -> CalculatedBondData:
    """
    Calculate dirty price, YTM, duration, and convexity for each bond at all pricing dates.

    Returns:
    A DataFrame containing YTM, duration, and convexity for each bond.
    """
    bond_description = x.bond_description
    clean_prices = x.clean_prices

    calculation_list = [bond_description[cusip].bond_calculation(clean_prices.xs(cusip, level='Identifier'))
                        for cusip in bond_description.keys()]
    
    calculated_data = pd.concat(calculation_list, axis=0, keys=bond_description.keys(), names=['Identifier', 'Pricing Date'])

    # Slower solution:
    # calculated_data = clean_prices.loc[valid_cusips].groupby('Identifier').apply(
    #     lambda x: dsc[x.name].bond_calculation(x.droplevel('Identifier'))) 
    return CalculatedBondData(calculated_data)


# Step 3: Calculate actual and proxied total bond returns
def approximate_total_return_index(contributions: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate bond returns using income, duration, and convexity contributions.

    Returns:
    A DataFrame containing the approximated returns.
    """
    income = contributions['Income Contribution']

    first_order_without_income = contributions['Duration Contribution']
    second_order_without_income = (contributions['Duration Contribution'] + contributions['Convexity Contribution'])

    first_order = (contributions['Income Contribution'] + first_order_without_income)
    second_order = (contributions['Income Contribution'] + second_order_without_income)

    approx_returns = pd.concat([income.rename('Income'),
                            first_order_without_income.rename('1st Order without Income'),
                            second_order_without_income.rename('2nd Order without Income'),
                            first_order.rename('1st Order'),
                            second_order.rename('2nd Order')], axis=1)
    
    def returns_accumulate(z: np.ndarray):
        acc = np.empty_like(z)
        acc[0, :] = 100.0  # Base value of 100
        acc[1:, :] = (1. + z[1:, :]).cumprod(axis=0) * 100.0

        return acc

    approx_tri = (approx_returns
                .groupby('Identifier')
                .apply(lambda x: pd.DataFrame(returns_accumulate(x.to_numpy()), 
                                                index=x.index.get_level_values('Pricing Date'),
                                                columns=approx_returns.columns))
                                                )
    
    return approx_tri


def _simple_return(x: pd.Series) -> float:
    """
    Calculate simple return from first to last value in a Series.
    
    Parameters:
    x (pd.Series): A Series of prices or index values.
    
    Returns:
    The simple return as a decimal.
    """
    return (x.iloc[-1] / x.iloc[0]) - 1.0


def _simple_return_helper(x: pd.Series) -> float:
    """
    Helper function to calculate simple return for groupby aggregation.
    Removes the 'Identifier' level from MultiIndex before calling _simple_return.
    
    Parameters:
    x (pd.Series): A Series with MultiIndex (Identifier, Pricing Date).
    
    Returns:
    The simple return as a decimal.
    """
    return _simple_return(x.droplevel('Identifier'))

 
class BondAnalysisResults:
    def __init__(self, contributions: pd.DataFrame, 
                 total_returns: pd.DataFrame, 
                 sample: pd.DataFrame,
                 approximate_tri: pd.DataFrame,
                 calculated_data: pd.DataFrame):
        self.contributions = contributions
        self.total_returns = total_returns
        self.sample = sample
        self.approximate_tri = approximate_tri
        self.calculated_data = calculated_data

    def subset(self, index, level):
        contributions = self.contributions.reindex(index, level=level)
        total_returns = self.total_returns.reindex(index, level=level)
        sample = self.sample.reindex(index, level=level)
        approximate_tri = self.approximate_tri.reindex(index, level=level)
        calculated_data = self.calculated_data.reindex(index, level=level)

        return BondAnalysisResults(contributions, total_returns, sample, approximate_tri, calculated_data)


    def approximation_stats(self) -> pd.Series:
        """
        Statistics on bond return contributions and approximation accuracy.
        """
        tr = self.total_returns
        abs_contributions = self.contributions.abs()

        return pd.Series({'Percent Income Contribution < Duration Contribution':
           (abs_contributions['Income Contribution'] < abs_contributions['Duration Contribution']).mean(),
           'Mean Income Contribution': abs_contributions['Income Contribution'].mean(),
           'Mean Duration Contribution': abs_contributions['Duration Contribution'].mean(), 
           'Mean Convexity Contribution': abs_contributions['Convexity Contribution'].mean(),
           'Percent 1st Order < Actual': (tr['1st Order'] < tr['Actual TR']).mean(), 
           'Min Actual - 1st Order': (tr['Actual TR'] - tr['1st Order']).min(),
           })
    

    def scatter_plot_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for scatter plots of actual vs approximated total returns.

        Returns:
        A DataFrame containing actual and approximated total returns for plotting.
        """
        tr = self.total_returns
        actual_tr = tr['Actual TR']
        approximate_tr = tr[['Income', '1st Order', '2nd Order', 
                             '1st Order without Income', '2nd Order without Income']]
        
        sample = self.sample

        return {k: pd.DataFrame({'Actual TR': actual_tr, 
                                 k: approximate_tr[k], 
                                 'Period (years)': sample['Return Period (years)']})
                                 
                                 for k in approximate_tr.columns}
    

    def clean_prices_stats(self, small_error_identifiers: pd.Index, large_error_identifiers: pd.Index) -> Dict[str, pd.DataFrame | pd.Series]:
        """
        Statistics on clean prices conditional on pricing errors.

        Returns:
        A DataFrame containing descriptive statistics on pricing errors.
        """
        clean_prices = self.calculated_data['Clean Price']

        all_ids = clean_prices.index.get_level_values(level='Identifier')

        small_error = clean_prices[all_ids.get_level_values('Identifier').isin(small_error_identifiers)]
        small_error_summary = pd.Series({'Min Clean Price': small_error.min(),
                                             'Max Clean Price': small_error.max()}
                                             )

        large_error = clean_prices[all_ids.get_level_values('Identifier').isin(large_error_identifiers)]
        g_large_error = large_error.groupby('Identifier')
        large_error_summary = pd.DataFrame({'Min Clean Price': g_large_error.min(),
                                            'Max Clean Price': g_large_error.max()}
                                            )
        
        return {'Large error summary': large_error_summary,
                'Small error summary': small_error_summary}


def return_comparison(x: CalculatedBondData) -> BondAnalysisResults:
    """
    Calculate actual and proxied bond returns.

    Returns:
    A BondAnalysisResults object containing the return contributions and the actual and approximated total returns.
    """
    calculated_data = x.data

    contributions = x.bond_return_decomposition()

    approximate_tri = approximate_total_return_index(contributions)

    actual_tr = (calculated_data.groupby('Identifier')['Total Return Index']  # Group actual TRI by bond identifier
                .aggregate(_simple_return_helper)  # Calculate annualized total return on each bond
                .rename('Actual TR'))

    approximate_tr = (approximate_tri.groupby('Identifier')  # Group approximated TRI by bond identifier
                    .aggregate(_simple_return_helper)  # Calculate annualized total return on each bond
                    )
    
    first_date = (calculated_data.groupby('Identifier')['Total Return Index']
                    .aggregate(lambda z: z.droplevel('Identifier').index[0])
                    .rename('First Sample Date')
                    )
    
    last_date = (calculated_data.groupby('Identifier')['Total Return Index']
                    .aggregate(lambda z: z.droplevel('Identifier').index[-1])
                    .rename('Last Sample Date')
                    )

    return_period = (((last_date - first_date).dt.days / 365.25)
                        .rename('Return Period (years)')
                        )

    sample = pd.concat([first_date, last_date, return_period], axis=1)

    total_returns = pd.concat([actual_tr, approximate_tr], axis=1)

    return BondAnalysisResults(contributions, total_returns, sample, approximate_tri, calculated_data)


def analysis_by_frequency(daily_calculated_bond_data: 'CalculatedBondData', 
                          sample_periods: Dict[str, int]) -> Dict:
    """
    Perform bond data analysis resampled to a specified frequency.

    Parameters:
    daily_calculated_bond_data (CalculatedBondData): Calculated bond data at business day frequency.
    sample_periods (Dict[str, int]): Dictionary mapping frequency strings to sample periods in business days.

    Returns:
    An AnalysisResults object containing the return contributions and the actual and approximated total returns.
    """
    dict_analysis_results = {}

    for k, v in sample_periods.items():
        resampled_data = daily_calculated_bond_data.resample(v)

        dict_analysis_results[k] = return_comparison(resampled_data)

    return dict_analysis_results
  

def scatter_plot_returns(x: Dict[str, pd.DataFrame], output_dir: Path | None=None):

    for k, v in x.items():
        fig, ax = plt.subplots()
        v.plot.scatter(ax=ax, x=0, y=1, c=2, cmap='Greys')

        # Add the y=x line
        ax.axline((0, 0), slope=1, linestyle='--', color='k')

        # Set uniform axis limits in all subplots
        lims = (
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_xlabel('Actual Total Return')
        ax.set_ylabel(k)
        
        if output_dir is not None:
            # Sanitize filename (replace spaces with underscores)
            filename = output_dir / (k.replace(' ', '_') + '.pdf')
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Saved plot: {filename}")


def scatter_plot_income_subperiods(calculated_bond_data: CalculatedBondData,
                                   output_dir: Optional[Path]=None):
    x1 = return_comparison(CalculatedBondData(calculated_bond_data.data.loc[pd.IndexSlice[:, '1980-01-01' : '1985-01-01'], :]))
    x2 = return_comparison(CalculatedBondData(calculated_bond_data.data.loc[pd.IndexSlice[:, '2013-01-01' : '2018-01-01'], :]))

    def approx_error(x: BondAnalysisResults):
        return (((1. + x.total_returns['2nd Order without Income']) / 
                 (1. + x.total_returns['Actual TR']) - 1.)
                .rename('2nd Order without Income'))
    
    e1 = approx_error(x1)
    e2 = approx_error(x2)
    
    fig, ax = plt.subplots()

    pd.concat((x1.sample, e1), axis=1).plot.scatter(x='Return Period (years)', 
                                                    y='2nd Order without Income',
                                                    ax=ax,
                                                    color='0.7')

    pd.concat((x2.sample, e2), axis=1).plot.scatter(x='Return Period (years)', y='2nd Order without Income',
                                                    ax=ax,
                                                    color='0.3')

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.))

    ax.legend(['1980 - 1985', '2013 - 2018'])
        
    if output_dir is not None:
        # Sanitize filename (replace spaces with underscores)
        filename = output_dir / 'Income_Subperiods.pdf'
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved plot: {filename}")



class PlotOptions:
    def __init__(self, yaxis_formatter: mtick.Formatter | None = None, 
                 ylim: tuple[float, float] | None = None,
                 ylabel: str | None = None) -> None:
        pass
        self.yaxis_formatter = yaxis_formatter
        self.ylim = ylim
        self.ylabel = ylabel


class Regime:
    def __init__(self, categories: pd.Index, values: pd.Series, bins: pd.IntervalIndex):
        self.categories = categories
        self.values = values
        self.bins = bins

    
    def apply_in_subperiods(self,
                            bond_analysis_results: BondAnalysisResults,
                            fun: Callable[[pd.api.typing.DataFrameGroupBy], pd.Series]) -> 'MetricsInSubperiods':
        """
        Apply a function to the analysis results of a bond
        in each sub-period where the regime is constant.
        """

        df = (pd.concat((bond_analysis_results.approximate_tri,
                         bond_analysis_results.contributions,
                         bond_analysis_results.calculated_data), axis=1)
              .join(self.values, on='Pricing Date', how='inner')
              .reset_index()
            )

        ttm_group = pd.cut(df['Time to Maturity'].dt.days / 365.25, 
                           [0, 1, 5, 10, 20, 30])
        
        subperiod_id = (df
                     .groupby('Identifier')['Regime']
                     .transform(lambda x: (x != x.shift())
                                .cumsum())
                     )
        
        grouper = (df
                   .assign(Subperiod_id=subperiod_id, Maturity_group=ttm_group)
                   .groupby(['Identifier', 'Subperiod_id'], observed=True)
                   )
        
        metrics_df = (grouper.agg(
            Regime=("Regime", "first"),
            Start=("Pricing Date", "first"),
            End=("Pricing Date", "last"),
            Initial_maturity_group=("Maturity_group", "first")
            )
            .assign(Period=lambda x: x["End"] - x["Start"])
            .assign(Apply=fun(grouper))
        )

        return MetricsInSubperiods(metrics_df, self)
    

    def diff_2nd_vs_1st_in_subperiods(self, 
                                      bond_analysis_results: BondAnalysisResults) -> 'MetricsInSubperiods':
        def fun(grouper):
            columns = ['1st Order', '2nd Order']
            return (grouper[columns].last() / grouper[columns].first()).diff(axis=1).abs().iloc[:, 1]

        return self.apply_in_subperiods(bond_analysis_results, fun)
    

    def diff_2nd_vs_2nd_wo_income_in_subperiods(self, 
                                                bond_analysis_results: BondAnalysisResults) -> 'MetricsInSubperiods':
        def fun(grouper):
            columns = ['2nd Order without Income', '2nd Order']
            return (grouper[columns].last() / grouper[columns].first()).diff(axis=1).abs().iloc[:, 1]

        return self.apply_in_subperiods(bond_analysis_results, fun)
    

    def apply_in_regime(self,
                        bond_analysis_results: BondAnalysisResults,
                        fun: Callable[[pd.api.typing.DataFrameGroupBy], pd.Series]) -> pd.DataFrame:
        """
        Apply a function to the analysis results of a bond per maturity group
        and regime.
        """
        
        df = (pd.concat((bond_analysis_results.approximate_tri,
                         bond_analysis_results.contributions,
                         bond_analysis_results.calculated_data), axis=1)
              .join(self.values, on='Pricing Date', how='inner')
              .reset_index()
            )

        ttm_group = pd.cut(df['Time to Maturity'].dt.days / 365.25, 
                           [0, 1, 5, 10, 20, 30])
        
        grouper = (df
                   .assign(Maturity_group=ttm_group)
                   .groupby(['Maturity_group', 'Regime'], observed=True)
                   )
        
        return (fun(grouper)
                .rename('Apply')
                .unstack('Regime')
                )
    

    def plot(self, df: pd.Series, options: Optional[PlotOptions]=None, output_fil: Optional[Path]=None):
        if options is None:
            options = PlotOptions()
    
        fig, ax = plt.subplots(figsize=(10, 4))

        df.plot(ax=ax, figsize=(10, 4), color='black')

        ax.grid(False)

        gray_levels = np.linspace(0.1, 0.7, len(self.categories))
        colors = np.column_stack([gray_levels] * 3)

        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_horizontalalignment("center")

        for interval, color in zip(self.bins, colors):
            ax.axhspan(interval.left, interval.right,
                       color=color, alpha=0.2)
        
        if options.yaxis_formatter is not None:
            ax.yaxis.set_major_formatter(options.yaxis_formatter)
            
        if output_fil is not None:
            fig.savefig(output_fil, bbox_inches='tight', dpi=300)


def quantile_regime(s: pd.Series, q: int, labels: Sequence):
    """
    Discretize a series into q regimes with the quantiles used as cutoff points.
    """
    x, y = pd.qcut(s, q=q, labels=labels, retbins=True)
    x = x.rename('Regime')
    y = pd.IntervalIndex.from_breaks(y) # type: ignore
    return Regime(x.dtype.categories, x, y) # type: ignore


class MetricsInSubperiods:
    def __init__(self, metrics: pd.DataFrame, regime: Regime) -> None:
        self.metrics = metrics
        self.regime = regime


    def average_by_regime(self):
        """
        Calculate the average of column "Apply" per regime and time spent in regime.
        """
        max_period = self.metrics['Period'].max()
        period_group = (pd.cut(self.metrics['Period'],
                           pd.TimedeltaIndex(['0d', '30d', '60d',
                                              '180d', '365d', max_period]))
                                              .rename('Period_group')
                                              )
        
        return (pd.concat((self.metrics, period_group), axis=1)
                    .groupby(['Regime', 'Period_group', 'Initial_maturity_group'], observed=False)['Apply']
                    .mean()
                    .unstack('Regime')
                    .reindex(self.regime.categories, axis=1)
                    )


def averages_by_regime(regime: Regime, bond_analysis_results: BondAnalysisResults):
    average_yield = regime.apply_in_regime(bond_analysis_results,
                                                 lambda x: x['Yield to Maturity'].mean())
    average_duration = regime.apply_in_regime(bond_analysis_results,
                                                    lambda x: x['Modified Duration'].mean())
    average_convexity = regime.apply_in_regime(bond_analysis_results,
                                                    lambda x: x['Modified Convexity'].mean())

    average_income_contribution = regime.apply_in_regime(bond_analysis_results,
                                                                lambda x: x['Income Contribution'].mean())
    average_duration_contribution = regime.apply_in_regime(bond_analysis_results,
                                                                lambda x: x['Duration Contribution'].apply(lambda x: x.abs().mean()))
    average_convexity_contribution = regime.apply_in_regime(bond_analysis_results,
                                                                lambda x: x['Convexity Contribution'].mean())
    
    return {'average_yield': average_yield * 1e2,
            'average_duration': average_duration,
            'average_convexity': average_convexity,
            'average_income_contribution': average_income_contribution * 1e3,
            'average_duration_contribution': average_duration_contribution * 1e4,
            'average_convexity_contribution': average_convexity_contribution * 1e6}


def averages_by_two_regimes_to_latex(
    left_results: Mapping[str, pd.DataFrame],
    right_results: Mapping[str, pd.DataFrame],
    path: Path,
    *,
    left_label: str = "Yield level regime",
    right_label: str = "Yield volatility regime",
    panel_labels: Mapping[str, str] | None = None,
    value_formatter: Callable[[float], str] | None = None,
) -> None:
    """
    Export two dictionaries of DataFrames side by side in one LaTeX table.

    Each dictionary must have the same keys. For each key, the corresponding
    DataFrames must have identical indexes and columns.

    The DataFrames are expected to have:
        - one interval index level;
        - three regime columns, such as low, medium, and high.
    """

    def format_maturity(interval: pd.Interval) -> str:
        return rf"{interval.left:g}--{interval.right:g}"

    def default_value_formatter(value: float) -> str:
        if pd.isna(value):
            return r"\multicolumn{1}{c}{-}"
        return f"{value:.1f}"

    if value_formatter is None:
        value_formatter = default_value_formatter

    left_keys = list(left_results)
    right_keys = list(right_results)

    if left_keys != right_keys:
        raise ValueError(
            "The two result dictionaries must have the same keys "
            "in the same order."
        )

    if not left_keys:
        raise ValueError("The result dictionaries cannot be empty.")

    first_left = left_results[left_keys[0]]

    if len(first_left.columns) != 3:
        raise ValueError(
            "This table layout expects exactly three regime columns."
        )

    # Validate all corresponding DataFrames.
    for key in left_keys:
        left = left_results[key]
        right = right_results[key]

        if not left.index.equals(right.index):
            raise ValueError(
                f"The DataFrames associated with {key!r} "
                "do not have identical indexes."
            )

        if not left.columns.equals(right.columns):
            raise ValueError(
                f"The DataFrames associated with {key!r} "
                "do not have identical columns."
            )

        if not left.columns.equals(first_left.columns):
            raise ValueError(
                f"The DataFrame associated with {key!r} "
                "does not have the reference columns."
            )

    column_labels = [
        str(column).title()
        for column in first_left.columns
    ]

    lines = [
        r"\begin{tabular}{l*{6}{d}}",
        r"\toprule",
        (
            r"Years to maturity"
            rf" & \multicolumn{{3}}{{c}}{{{left_label}}}"
            rf" & \multicolumn{{3}}{{c}}{{{right_label}}} \\"
        ),
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        (
            r"& "
            + " & ".join(
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{label}}}}}"
                for label in column_labels
            )
            + " & "
            + " & ".join(
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{label}}}}}"
                for label in column_labels
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    for panel_number, key in enumerate(left_keys):
        left = left_results[key]
        right = right_results[key]

        if panel_labels is None:
            panel_label = key.replace("_", " ").title()
        else:
            panel_label = panel_labels.get(
                key,
                key.replace("_", " ").title(),
            )

        lines.append(
            rf"\multicolumn{{7}}{{l}}{{\textbf{{{panel_label}}}}} \\"
        )

        for maturity in left.index:
            maturity_label = format_maturity(maturity)

            left_values = " & ".join(
                value_formatter(value)
                for value in left.loc[maturity]
            )

            right_values = " & ".join(
                value_formatter(value)
                for value in right.loc[maturity]
            )

            lines.append(
                rf"{maturity_label} & "
                rf"{left_values} & {right_values} \\"
            )

        if panel_number < len(left_keys) - 1:
            lines.extend([
                r"\addlinespace",
                r"\cmidrule(lr){2-7}",
                r"\addlinespace",
            ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def by_regime_2_levels_to_latex(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Export DataFrame to LaTeX format, assuming two index levels:
        1. Period_group: Timedelta intervals
        2. Initial_maturity_group: numerical intervals
    """

    def format_period(interval: pd.Interval) -> str:
        left = interval.left.days
        right = interval.right.days
        return rf"{left}--{right} days"

    def format_maturity(interval: pd.Interval) -> str:
        left = interval.left
        right = interval.right
        return rf"{left:g}--{right:g}"

    def format_value(value: float) -> str:
        if pd.isna(value):
            return r"\multicolumn{1}{c}{-}"
        return f"{value:.1f}"

    table = df.rename(str.title, axis="columns")
    n_columns = len(table.columns)

    if n_columns != 3:
        raise ValueError(
            "This table layout expects exactly three data columns."
        )

    lines = [
        r"\begin{tabular}{ll*{3}{d}}",
        r"\toprule",
        (
            r"Days in regime & Years to maturity & "
            rf"\multicolumn{{3}}{{c}}{{{df.columns.name}}} \\"
        ),
        r"\cmidrule(lr){2-5}",
        (
            r"& & "
            + " & ".join(
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{column}}}}}"
                for column in table.columns
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    grouped = table.groupby(
        level="Period_group",
        sort=False,
        observed=True,
    )

    groups = list(grouped)

    for group_number, (period, panel) in enumerate(groups):
        period_label = format_period(period)

        # Separate heading row at the top of the panel.
        lines.append(
            rf"\multicolumn{{2}}{{l}}{{{period_label}}}"
            + r" & \multicolumn{3}{c}{} \\"
        )

        for index, row in panel.iterrows():
            maturity = index[
                table.index.names.index("Initial_maturity_group")
            ]
            maturity_label = format_maturity(maturity)

            values = " & ".join(
                format_value(value)
                for value in row
            )

            lines.append(
                rf"& {maturity_label} & {values} \\"
            )

        # Rule between panels, restricted to the numerical columns.
        if group_number < len(groups) - 1:
            lines.append(r"\cmidrule(lr){2-5}")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def by_regime_panels_to_latex(
    dataframes: Sequence[pd.DataFrame],
    labels: Sequence[str],
    path: Path,
) -> None:
    """
    Export several DataFrames as vertically stacked panels in one LaTeX
    tabular environment.

    All DataFrames must:
        - have one interval index level;
        - have exactly three columns;
        - have identical row and column labels.

    Parameters
    ----------
    dataframes:
        DataFrames to display, in panel order.
    labels:
        Panel headings, in the same order as `dataframes`.
    path:
        Destination of the generated LaTeX file.
    """

    def format_maturity(interval: pd.Interval) -> str:
        left = interval.left
        right = interval.right
        return rf"{left:g}--{right:g}"

    def format_value(value: float) -> str:
        if pd.isna(value):
            return r"\multicolumn{1}{c}{-}"
        return f"{value:.1f}"

    if not dataframes:
        raise ValueError("At least one DataFrame must be provided.")

    if len(dataframes) != len(labels):
        raise ValueError(
            "There must be exactly one label for each DataFrame."
        )

    reference = dataframes[0]

    if len(reference.columns) != 3:
        raise ValueError(
            "This table layout expects exactly three data columns."
        )

    for panel_number, dataframe in enumerate(dataframes, start=1):
        if not dataframe.index.equals(reference.index):
            raise ValueError(
                f"DataFrame {panel_number} does not have the reference index."
            )

        if not dataframe.columns.equals(reference.columns):
            raise ValueError(
                f"DataFrame {panel_number} does not have the "
                "reference columns."
            )

    tables = [
        dataframe.rename(str.title, axis="columns")
        for dataframe in dataframes
    ]

    column_group_name = reference.columns.name or ""

    lines = [
        r"\begin{tabular}{l*{3}{d}}",
        r"\toprule",
        (
            r"Years to maturity & "
            rf"\multicolumn{{3}}{{c}}{{{column_group_name}}} \\"
        ),
        r"\cmidrule(lr){2-4}",
        (
            r"& "
            + " & ".join(
                rf"\multicolumn{{1}}{{c}}{{\textbf{{{column}}}}}"
                for column in tables[0].columns
            )
            + r" \\"
        ),
        r"\midrule",
    ]

    for panel_number, (label, table) in enumerate(
        zip(labels, tables, strict=True)
    ):
        lines.append(
            rf"\multicolumn{{4}}{{l}}{{\textbf{{{label}}}}} \\"
        )

        for maturity, row in table.iterrows():
            maturity_label = format_maturity(maturity)

            values = " & ".join(
                format_value(value)
                for value in row
            )

            lines.append(
                rf"{maturity_label} & {values} \\"
            )

        if panel_number < len(tables) - 1:
            lines.extend([
                r"\addlinespace",
                r"\cmidrule(lr){2-4}",
                r"\addlinespace",
            ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    