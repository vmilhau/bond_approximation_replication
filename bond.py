from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from numba import njit
from pandas._libs.missing import NAType


# To create the cash flow schedule of a bond, we assume that the maximum maturity at issuance is 40 years
MAX_YEARS_TO_MATURITY = 40


# ===== JIT-OPTIMIZED FUNCTIONS =====

@njit
def pv_jit(rate, cash_flows: np.ndarray, fraction_coupon_period_ahead: float) -> float:
    """
    Present Value (PV) calculation using discrete compounding for discounting.
    Optimized for numba JIT compilation.
    """
    pv = 0.0
    for t in range(len(cash_flows)):
        pv += cash_flows[t] / (1. + rate)**(t + fraction_coupon_period_ahead)
    return pv


@njit
def bisection(a, b, cash_flows: np.ndarray, fraction_coupon_period_ahead: float, price: float, tol: float=1e-12, maxiter: int=100) -> float:
    """
    Bisection method to find root of pv_jit function.
    Optimized for numba JIT compilation.
    """
    fa = pv_jit(a, cash_flows, fraction_coupon_period_ahead) - price
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = pv_jit(m, cash_flows, fraction_coupon_period_ahead) - price
        if (abs(fm) < tol) or (0.5 * (b - a) < tol):
            return m
        if fa * fm < 0:
            b = m
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


@njit
def total_return_index_jit(total_returns: np.ndarray, base_value: float) -> np.ndarray:
    """
    Total Return Index calculation.
    Optimized for numba JIT compilation.
    """
    tri = np.empty(len(total_returns) + 1)
    tri[0] = base_value
    tri[1:] = base_value * np.cumprod(total_returns)
    return tri


@njit
def modified_duration_jit(cash_flows: np.ndarray, fraction_coupon_period_ahead: float, rate: float, dirty_price: float):
    """
    Modified duration calculation.
    Optimized for numba JIT compilation.
    """    
    s = 0.0
    for t in range(len(cash_flows)):  # Loop is faster than vectorized version in numba because of overhead of creating temporary arrays
        s += (cash_flows[t] * (t + fraction_coupon_period_ahead)
              / (1. + rate)**(t + fraction_coupon_period_ahead + 1.)
              )

    if dirty_price == 0.0:
        modified_duration = np.nan
    else:
        modified_duration = s / dirty_price

    return modified_duration


@njit
def modified_convexity_jit(cash_flows: np.ndarray, fraction_coupon_period_ahead: float, rate: float, dirty_price: float):
    """
    Modified convexity calculation.
    Optimized for numba JIT compilation.
    """    
    s = 0.0
    for t in range(len(cash_flows)):
        s += (cash_flows[t] * (t + fraction_coupon_period_ahead) * (t + fraction_coupon_period_ahead + 1.) 
              / (1. + rate)**(t + fraction_coupon_period_ahead + 2.)
              )

    if dirty_price == 0.0:
        modified_convexity = np.nan
    else:
        modified_convexity = s / dirty_price

    return modified_convexity


# ===== ABSTRACT BASE CLASS =====

class Bond(ABC):
    """
    Abstract base class for bonds.
    """
    _cf_dates: np.ndarray
    _cf_values: np.ndarray
    maturity_date: pd.Timestamp


    def time_to_maturity(self, date: pd.Timestamp) -> pd.Timedelta:
        """
        Calculate the residual time to maturity of the bond in days.
        """
        return self.maturity_date - date


    @abstractmethod
    def ytm(self, dirty_price: float, date: pd.Timestamp) -> float:
        """
        Calculate the Yield to Maturity (YTM) of the bond given its dirty price.

        Parameters:
        dirty_price (float): The current dirty price of the bond.
        date (pd.Timestamp): The date at which to calculate the YTM.

        Returns:
        The YTM as a decimal, discretely compounded, in annual terms.
        """
        pass

    @abstractmethod
    def log_price_time_derivative(self, ytm: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the logarithm of the bond price with respect to time.

        Parameters:
        ytm (np.ndarray): The YTM values.

        Returns:
        An array of the derivative values.
        """
        pass


    @abstractmethod
    def modified_duration(self, date: pd.Timestamp, dirty_price: float, ytm: float) -> float:
        """
        Calculate the Modified Duration of the bond.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the duration.
        dirty_price (float): The current dirty price of the bond.
        ytm (float): The Yield to Maturity as a decimal.

        Returns:
        The Modified Duration as a decimal, in years.
        """
        pass


    @abstractmethod
    def modified_convexity(self, date: pd.Timestamp, dirty_price: float, ytm: float) -> float:
        """
        Calculate the Modified Convexity of the bond.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the convexity.
        market_price (float): The current dirty price of the bond.
        ytm (float): The Yield to Maturity as a decimal.

        Returns:
        The Modified Convexity as a decimal, in years squared.
        """
        pass


    @abstractmethod
    def clean2dirty_prices(self, clean_prices: pd.Series) -> pd.Series:
        """
        Convert clean prices to dirty prices by adding accrued interest if applicable.

        Parameters:
        clean_prices (pd.Series): The clean prices of the bond.

        Returns:
        The dirty prices of the bond.
        """
        pass


    def total_return_index(self, dirty_prices: pd.Series) -> pd.Series:
        """
        Calculate the Total Return Index of the bond over a series of dates.

        Parameters:
        dirty_prices (pd.Series): A Series of dirty prices indexed by date.

        Returns:
        A Series containing the Total Return Index values for each date.
        """
        cash_flows = self._cf_values

        mask = (self._cf_dates >= dirty_prices.index[0]) \
                & (self._cf_dates <= dirty_prices.index[-1])
        cf_dates = self._cf_dates[mask]

        cash_flows_ext = np.zeros(len(dirty_prices))
        cash_flows_ext[np.isin(dirty_prices.index, cf_dates)] = cash_flows[mask]

        dirty_prices_numpy = dirty_prices.to_numpy()

        total_returns = (dirty_prices_numpy + cash_flows_ext)[1:] / dirty_prices_numpy[:-1]

        total_return_index = pd.Series(total_return_index_jit(total_returns, 100.), index=dirty_prices.index)

        return total_return_index


    def bond_calculation(self, clean_prices: pd.Series) -> pd.DataFrame:
        """
        Calculate dirty prices, total return indices, YTM, modified duration, and modified convexity.
        """
        dirty_prices = self.clean2dirty_prices(clean_prices)
        total_return_index = self.total_return_index(dirty_prices)

        ytm = np.array([self.ytm(dpr, t) for t, dpr in dirty_prices.items()])

        log_price_time_derivative = self.log_price_time_derivative(ytm)

        time_to_maturity = np.array([self.time_to_maturity(t) for t in dirty_prices.index])

        modified_duration = np.array([self.modified_duration(t, dpr, y) 
                                      for t, dpr, y in zip(dirty_prices.index, dirty_prices, ytm)])
        
        modified_convexity = np.array([self.modified_convexity(t, dpr, y) 
                                       for t, dpr, y in zip(dirty_prices.index, dirty_prices, ytm)])

        return pd.DataFrame({'Clean Price': clean_prices,
                            'Dirty Price': dirty_prices,
                            'Total Return Index': total_return_index,
                            'Time to Maturity': time_to_maturity,
                            'Yield to Maturity': ytm,
                            'Log Price Time Derivative': log_price_time_derivative,
                            'Modified Duration': modified_duration,
                            'Modified Convexity': modified_convexity})


# ===== BOND IMPLEMENTATIONS =====

class ZeroCouponBond(Bond):
    """
    Represent a zero-coupon bond with a maturity date.
    """

    def __init__(self, 
                 face_value: float, 
                 maturity_date: pd.Timestamp, 
                 issue_dates: pd.DatetimeIndex, 
                 cusip: str):
            self.face_value = float(face_value)
            self.maturity_date = maturity_date
            self.issue_dates = issue_dates
            self.cusip = cusip

            self._cf_dates = np.busday_offset([self.maturity_date.to_numpy().astype('datetime64[D]')], 0, 'modifiedfollowing')
            self._cf_values = np.array([self.face_value])


    def __str__(self):
        issue_dates_string = ", ".join(
            "NaT" if pd.isna(t) else t.strftime("%Y-%m-%d")
            for t in self.issue_dates
        )

        return (f"ZeroCouponBond(CUSIP={self.cusip}, Face Value={self.face_value}, "
                f"Maturity Date={self.maturity_date.date()}, "
                f"Issue Dates={issue_dates_string})")
    
    
    def __repr__(self):
        return self.__str__()
    

    def time_to_maturity(self, date: pd.Timestamp) -> pd.Timedelta:
        return self._cf_dates[-1] - date
    

    def ytm(self, dirty_price: float, date: pd.Timestamp) -> float:
        """
        Calculate the Yield to Maturity (YTM) of the zero-coupon bond given its market price.
        YTM is calculated using continuous compounding.

        Parameters:
        price (float): The current price of the bond.
        date (pd.Timestamp): The date at which to calculate the YTM.

        Returns:
        The YTM as a decimal, annually compounded and in annual terms.
        """
        if (date > self._cf_dates[-1]) or (date < self.issue_dates[0]):
            return np.nan
        
        if date == self._cf_dates[-1]:
            return 0.0
        
        time_to_last_cf = (self._cf_dates[-1] - date).days / 365.25

        ytm = (self.face_value / dirty_price)**(1. / time_to_last_cf) - 1.
        return ytm
    

    def log_price_time_derivative(self, ytm: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the logarithm of the bond price with respect to time.

        Parameters:
        ytm (np.ndarray): The YTM values.

        Returns:
        An array of the derivative values.
        """
        return np.log(1. + ytm)


    def modified_duration(self, date: pd.Timestamp, dirty_price: float=0., ytm: float=0.) -> float:
        """
        Calculate the Modified Duration of the zero-coupon bond.
        Continuous compounding is assumed.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the duration.
        market_price (float): Not used for zero-coupon bonds but kept for interface consistency.
        ytm (float): The YTM, annually compounded and in annual terms.

        Returns:
        The Modified Duration as a decimal.
        """
        years_to_maturity = (self._cf_dates[-1] - date).days / 365.25

        return years_to_maturity / (1. + ytm)


    def modified_convexity(self, date: pd.Timestamp, dirty_price: float=0., ytm: float=0.) -> float:
        """
        Calculate the Modified Convexity of the zero-coupon bond.
        Continuous compounding is assumed.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the convexity.
        market_price (float): Not used for zero-coupon bonds but kept for interface consistency.
        ytm (float): Not used for zero-coupon bonds but kept for interface consistency.

        Returns:
        The Modified Convexity as a decimal.
        """
        years_to_maturity = (self._cf_dates[-1] - date).days / 365.25

        return years_to_maturity * (years_to_maturity + 1.) / (1. + ytm)**2


    def clean2dirty_prices(self, clean_prices: pd.Series) -> pd.Series:
        """
        Calculate dirty prices from clean prices for zero-coupon bonds.
        Since zero-coupon bonds do not accrue interest, dirty prices equal clean prices.
        """
        dirty_prices = clean_prices.copy().rename('Dirty Price')
        if self._cf_dates[-1] in dirty_prices.index:
            dirty_prices.loc[self._cf_dates[-1]] = 0.0  # At the last cash flow date, price falls to zero
        return dirty_prices
    

class CouponBond(Bond):
    """
    Represent a fixed-income bond with periodic interest payments and a maturity date.
    """

    def __init__(self, 
                 face_value: float, 
                 maturity_date: pd.Timestamp, 
                 coupon_rate_pct: float, 
                 coupon_frequency: str,
                 issue_dates: pd.DatetimeIndex, 
                 cusip: str):
            self.face_value = float(face_value)
            self.maturity_date = maturity_date
            self.coupon_rate_pct = float(coupon_rate_pct)
            self.coupon_frequency = coupon_frequency
            self.issue_dates = issue_dates
            self.cusip = cusip

            periods_per_year = self.periods_per_year()
            q = 12 // periods_per_year
        
            coupon_payment = (self.coupon_rate_pct / 100.) * self.face_value / periods_per_year

            def coupon_dates_gen():
                k = 0  # Number of coupon payments before redemption
                full_years_to_maturity = k // periods_per_year
                t = self.maturity_date

                while not(t < self.issue_dates[0]) and (full_years_to_maturity <= MAX_YEARS_TO_MATURITY):
                    yield t
                    k += 1
                    full_years_to_maturity = k // periods_per_year
                    t = self.maturity_date - pd.DateOffset(months=k * q)

            payment_dates = pd.DatetimeIndex([t for t in coupon_dates_gen()])[::-1]

            # Add first quasi-coupon date before issuance date
            if payment_dates[0] > self.issue_dates[0]:
                payment_dates = payment_dates.insert(0, payment_dates[0] - pd.DateOffset(months=q))

            payments = [0.0] + [coupon_payment] * (len(payment_dates) - 2) + [coupon_payment + self.face_value]
            self._cf_dates = np.busday_offset(payment_dates.to_numpy().astype('datetime64[D]'), 0, 'modifiedfollowing')
            self._cf_values = np.array(payments)

    def __str__(self):
        issue_dates_string = ", ".join(
            "NaT" if pd.isna(t) else t.strftime("%Y-%m-%d")
            for t in self.issue_dates
        )

        return (f"CouponBond(CUSIP={self.cusip}, Face Value={self.face_value}, "
                f"Maturity Date={self.maturity_date.date()}, "
                f"Coupon Rate={self.coupon_rate_pct}%, "
                f"Coupon Frequency={self.coupon_frequency}, "
                f"Issue Date(s)={issue_dates_string})")
    

    def __repr__(self):
        return self.__str__()
    

    def periods_per_year(self) -> int:
        """
        Determine the number of coupon payment periods per year based on the coupon frequency.

        Returns:
        The number of periods per year as an integer.
        """
        match self.coupon_frequency:
            case '1':
                return 1
            case 'S' | '2' | 'C':
                return 2
            case 'Q' | '3':
                return 4
            case 'M':
                return 12
            case _:
                raise ValueError(f"Unsupported coupon frequency: {self.coupon_frequency}")


    def fraction_coupon_period_ahead(self, date: pd.Timestamp) -> float:
        """
        Calculate the fraction of coupon period ahead.
        """
        if (date > self._cf_dates[-1]) or (date < self.issue_dates[0]):
            return np.nan
        
        if (date == self._cf_dates[-1]):
            return 0.0
        
        date_numpy = date.to_numpy()
        
        i = np.searchsorted(self._cf_dates, date_numpy, side='right')

        return ((self._cf_dates[i] - date_numpy).astype('timedelta64[D]').astype(float) /
                (self._cf_dates[i] - self._cf_dates[i - 1]).astype('timedelta64[D]').astype(float)
                )


    def ytm(self, dirty_price: float, date: pd.Timestamp) -> float:
        """
        Calculate the Yield to Maturity (YTM) of the bond given its dirty price.
        YTM is calculated using discrete compounding, with compounding frequency equal to the coupon frequency.

        Parameters:
        market_price (float): The current market price of the bond.
        date (pd.Timestamp): The date at which to calculate the YTM.

        Returns:
        The YTM as a decimal.
        """
        if (date > self._cf_dates[-1]) or (date < self.issue_dates[0]):
            return np.nan
        
        if date == self._cf_dates[-1]:
            return 0.0

        date_numpy = date.to_numpy()
        cash_flows = self._cf_values[self._cf_dates > date_numpy]
        fraction_coupon_period_ahead = self.fraction_coupon_period_ahead(date)

        rate = bisection(-0.99, 1.0, cash_flows, fraction_coupon_period_ahead, dirty_price)
        return rate * self.periods_per_year()
    

    def log_price_time_derivative(self, ytm: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the logarithm of the bond price with respect to time.

        Parameters:
        ytm (np.ndarray): The YTM values.

        Returns:
        An array of the derivative values.
        """
        q = self.periods_per_year()
        return q * np.log(1. + ytm / q)


    def modified_duration(self, date: pd.Timestamp, dirty_price: float, ytm: float) -> float:
        """
        Calculate the Modified Duration of the bond.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the duration.
        dirty_price (float): The current dirty price of the bond.
        ytm (float): The Yield to Maturity as a decimal, discretely compounded, in annual terms.

        Returns:
        The Modified Duration as a decimal, in years.
        """
        if (date > self._cf_dates[-1]) or (date < self.issue_dates[0]):
            return np.nan
        
        if date == self._cf_dates[-1]:
            return 0.0

        date_numpy = date.to_numpy()
        cash_flows = self._cf_values[self._cf_dates > date_numpy]
        fraction_coupon_period_ahead = self.fraction_coupon_period_ahead(date)

        rate = ytm / self.periods_per_year()

        modified_duration = (modified_duration_jit(cash_flows, fraction_coupon_period_ahead, rate, dirty_price)
                             / self.periods_per_year()
                             )

        return modified_duration


    def modified_convexity(self, date: pd.Timestamp, dirty_price: float, ytm: float) -> float:
        """
        Calculate the Modified Convexity of the bond.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the convexity.
        dirty_price (float): The current dirty price of the bond.
        ytm (float): The Yield to Maturity as a decimal, discretely compounded, in annual terms.

        Returns:
        The Modified Convexity as a decimal, in years squared.
        """
        if (date > self._cf_dates[-1]) or (date < self.issue_dates[0]):
            return np.nan
        
        if date == self._cf_dates[-1]:
            return 0.0

        date_numpy = date.to_numpy()
        cash_flows = self._cf_values[self._cf_dates > date_numpy]
        fraction_coupon_period_ahead = self.fraction_coupon_period_ahead(date)

        rate = ytm / self.periods_per_year()

        modified_convexity = (modified_convexity_jit(cash_flows, fraction_coupon_period_ahead, rate, dirty_price)
                              / self.periods_per_year()**2
                              )

        return modified_convexity
    

    def accrued_interest(self, date: pd.Timestamp) -> float:
        """
        Calculate the accrued interest of the bond as of a given date.

        Parameters:
        date (pd.Timestamp): The date at which to calculate the accrued interest.

        Returns:
        The accrued interest amount.
        """
        coupon_payment = (self.coupon_rate_pct / 100.) * self.face_value / self.periods_per_year()
        
        accrued_interest = coupon_payment * (1. - self.fraction_coupon_period_ahead(date))

        return accrued_interest
    

    def accrued_interest_vectorized(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Fast calculation of accrued interest at multiple dates.

        Parameters:
        dates (pd.DatetimeIndex): A Series of dates at which to calculate the accrued interest.

        Returns:
        A Series containing the accrued interest amounts for each date.
        """
        accrued_interests = np.zeros_like(dates, dtype=float)

        mask_intermediate = (dates >= self._cf_dates[0]) & (dates < self._cf_dates[-1])

        dates_intermediate = dates[mask_intermediate].to_numpy().astype('datetime64[D]')

        rep_cf_dates = np.tile(self._cf_dates, (len(dates_intermediate), 1))
        rep_dates = np.tile(dates_intermediate.reshape(-1, 1), (1, len(self._cf_dates)))

        maturities = rep_dates - rep_cf_dates
        time_from_previous_cf = maturities[:, :-1]
        time_to_next_cf = maturities[:, 1:]
        mask = (time_from_previous_cf >= pd.Timedelta(0.)) & (time_to_next_cf < pd.Timedelta(0.))
        # There is exactly one True value per row (i.e., per date)

        coupon_payment = (self.coupon_rate_pct / 100.) * self.face_value / self.periods_per_year()

        accrued_interests[mask_intermediate] = \
            time_from_previous_cf[mask] / (time_from_previous_cf[mask] - time_to_next_cf[mask]) * coupon_payment

        return accrued_interests
    

    def clean2dirty_prices(self, clean_prices: pd.Series) -> pd.Series:
        """
        Calculate dirty prices from clean prices by adding accrued interest.
        """
        ai = self.accrued_interest_vectorized(clean_prices.index)
        dirty_prices = (clean_prices + ai).rename('Dirty Price')
        if self._cf_dates[-1] in dirty_prices.index:
            dirty_prices.loc[self._cf_dates[-1]] = 0.0  # At the last cash flows date, price falls to zero
        return dirty_prices
