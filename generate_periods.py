import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Union, List, Optional
import pandas_market_calendars as mcal

def add_trading_days(
    dates: Union[pd.Series, pd.Index, List[str], pd.DatetimeIndex],
    horizon: int,
    trading_days: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """
    Add N trading days to each date using a trading calendar.
    Inputs:
      - dates: List of dates to be added
      - horizon (int): Number of trading days to add
      - trading_days: List of trading days in the NYSE universe
    Outputs:
      - List of dates after addition
    """

    idxs = np.searchsorted(trading_days, dates, side="left") + horizon
    result = trading_days[np.clip(idxs, 0, len(trading_days)-1)]
    return pd.to_datetime(result)

def generate_full_periods_train_test_valid(
        train_start_date: str,
        train_end_date: str,
        test_period_start_offset_months: int,
        test_period_months: int,
        validation_period_months: int,
        validation_offset_months: int=0,
        prediction_horizon: int=7,
        trading_calendar: str='NYSE',
        trading_dates: Optional[pd.DatetimeIndex]=None,
    ):
    """
    Generate full periods for training, validation, and testing.
    Args:
        train_start_date (str): Start date for training period in 'YYYY-MM-DD'
        train_end_date (str): End date for training period in 'YYYY-MM-DD'
        test_period_months (int): Duration of the test period in months
        validation_period_months (int): Duration of the validation period in months
        trading_dates (pd.DatetimeIndex): Trading calendar dates
        validation_offset_months (int): Offset months before validation starts
    Returns:
        List[str]: List containing start and end dates for training, validation, and testing periods.
    """

    # Convert to datetime
    start_dt = datetime.strptime(train_start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(train_end_date, '%Y-%m-%d')
    
    # Validation
    internal_train_start = start_dt
    internal_train_end = end_dt - relativedelta(months=validation_period_months+validation_offset_months)
    validation_start_date = internal_train_end + relativedelta(months=validation_offset_months)
    validation_end_date = end_dt

    # Test
    if test_period_start_offset_months > 0:
        test_start_date = end_dt + relativedelta(months=test_period_start_offset_months)
        test_end_date = test_start_date + relativedelta(months=test_period_months)
    else:
        if trading_dates is None:
            nyse = mcal.get_calendar(trading_calendar)
            schedule = nyse.schedule(start_date='2014-01-01', end_date='2030-12-31')
            trading_dates = schedule.index

        test_start_date = add_trading_days(
            pd.to_datetime([train_end_date]),
            horizon=prediction_horizon,
            trading_days=trading_dates
        )[0]
        test_end_date = test_start_date + relativedelta(months=test_period_months)

    # Format
    train_start = start_dt.strftime('%Y-%m-%d')
    train_end = end_dt.strftime('%Y-%m-%d')
    internal_train_start = internal_train_start.strftime('%Y-%m-%d')
    internal_train_end = internal_train_end.strftime('%Y-%m-%d')
    validation_start_date = validation_start_date.strftime('%Y-%m-%d')
    validation_end_date = validation_end_date.strftime('%Y-%m-%d')
    test_start_date = test_start_date.strftime('%Y-%m-%d')
    test_end_date = test_end_date.strftime('%Y-%m-%d')

    return [
        train_start, train_end,
        internal_train_start, internal_train_end,
        validation_start_date, validation_end_date,
        test_start_date, test_end_date
    ]


def generate_test_periods(start, end):
    import pandas as pd
    periods = []
    # start, end = '2017-12-31', '2023-09-30'
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    # Move current_date to the 1st of the next month after start_date
    current_date = (start_date + pd.offsets.MonthBegin(1))

    while current_date < end_date:
        # Train: from previous month's end to one year later
        train_start = (current_date - pd.offsets.Day(1)).strftime('%Y-%m-%d')
        train_end = (current_date + pd.DateOffset(years=5, months=3) - pd.offsets.Day(1)).strftime('%Y-%m-%d')

        # Test: from the 1st of next month after training ends to that month's end
        test_start = (current_date + pd.DateOffset(years=5, months=4)).strftime('%Y-%m-%d')
        test_end = (pd.to_datetime(test_start) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        periods.append((train_start, train_end, test_start, test_end))

        # move to next month's 1st day
        current_date += pd.offsets.MonthBegin(1)

    return periods

def get_train_periods(train_start, train_end, config):
    dyn = generate_test_periods('2015-03-31', '2025-12-30')
    dyn_train_periods = [[p[0], p[1]] for p in dyn]

    prev = [
        ['2015-03-31', '2015-12-31'],
        ['2015-03-31', '2016-01-31'],
        ['2015-03-31', '2016-02-29'],
        ['2015-03-31', '2016-03-31'],
        ['2015-03-31', '2016-04-30'],
        ['2015-03-31', '2016-05-31'],
        ['2015-03-31', '2016-06-30'],
        ['2015-03-31', '2016-07-31'],
        ['2015-03-31', '2016-08-31'],
        ['2015-03-31', '2016-09-30'],
        ['2015-03-31', '2016-10-31'],
        ['2015-03-31', '2016-11-30'],
        ['2015-03-31', '2016-12-31'],
        ['2015-03-31', '2017-01-31'],
        ['2015-03-31', '2017-02-28'],
        ['2015-03-31', '2017-03-31'],
        ['2015-03-31', '2017-04-30'],
        ['2015-03-31', '2017-05-31'],
        ['2015-03-31', '2017-06-30'],
        ['2015-03-31', '2017-07-31'],
        ['2015-03-31', '2017-08-31'],
        ['2015-03-31', '2017-09-30'],
        ['2015-03-31', '2017-10-31'],
        ['2015-03-31', '2017-11-30'],
        ['2015-03-31', '2017-12-31'],
        ['2015-03-31', '2018-01-31'],
        ['2015-03-31', '2018-02-28'],
        ['2015-03-31', '2018-03-31'],
        ['2015-03-31', '2018-04-30'],
        ['2015-03-31', '2018-05-31'],
        ['2015-03-31', '2018-06-30'],
        ['2015-03-31', '2018-07-31'],
        ['2015-03-31', '2018-08-31'],
        ['2015-03-31', '2018-09-30'],
        ['2015-03-31', '2018-10-31'],
        # 2019
        ['2015-03-31', '2018-11-30'],
        ['2015-03-31', '2018-12-31'],
        ['2015-03-31', '2019-01-31'],
        ['2015-03-31', '2019-02-28'],
        ['2015-03-31', '2019-03-31'],
        ['2015-03-31', '2019-04-30'],
        ['2015-03-31', '2019-05-31'],
        ['2015-03-31', '2019-06-30'],
        ['2015-03-31', '2019-07-31'],
        ['2015-03-31', '2019-08-31'],
        ['2015-03-31', '2019-09-30'],
        ['2015-03-31', '2019-10-31'],
        ['2015-03-31', '2019-11-30'],
        ['2015-03-31', '2019-12-31'],
        ['2015-03-31', '2020-01-31'],
        ['2015-03-31', '2020-02-29'],
        ['2015-03-31', '2020-03-31'],
        ['2015-03-31', '2020-04-30'],
        ['2015-03-31', '2020-05-31'],
    ]

    # Combined list is sorted ascending by train_end (prev tops out at 2020-05-31,
    # dynamic starts at 2020-06-30), enabling binary-search pre-filtering.
    all_base_periods = prev + dyn_train_periods

    offset      = config.experiment.test_period_start_offset_months
    test_months = config.experiment.test_period_months
    horizon     = config.experiment.prediction_horizon

    train_start_dt = datetime.strptime(train_start, '%Y-%m-%d')
    train_end_dt   = datetime.strptime(train_end,   '%Y-%m-%d')

    # Compute bounds on train_end_date that can possibly produce a test window
    # overlapping [train_start, train_end], then bisect the sorted list so we
    # only call generate_full_periods_train_test_valid on candidates.
    if offset > 0:
        # test_start = train_end_date + offset months
        # test_end   = test_start + test_months months
        # need: test_end >= train_start  →  train_end_date >= train_start - offset - test_months
        # need: test_start <= train_end  →  train_end_date <= train_end - offset
        lo = train_start_dt - relativedelta(months=offset + test_months)
        hi = train_end_dt   - relativedelta(months=offset)
        trading_dates = None
    else:
        # test_start = train_end_date + H trading days  (~H*1.5 calendar days max)
        # need: test_end >= train_start  →  train_end_date >= train_start - test_months - slack
        # need: test_start <= train_end  →  train_end_date <= train_end  (safe upper bound)
        slack = timedelta(days=int(horizon * 1.5) + 5)
        lo = train_start_dt - relativedelta(months=test_months) - slack
        hi = train_end_dt

        # Load trading calendar once for all candidate calls.
        nyse = mcal.get_calendar(config.experiment.trading_calendar)
        trading_dates = nyse.schedule(start_date='2014-01-01', end_date='2030-12-31').index

    lo_str = lo.strftime('%Y-%m-%d')
    hi_str = hi.strftime('%Y-%m-%d')

    train_ends = [p[1] for p in all_base_periods]
    left  = bisect_left(train_ends,  lo_str)
    right = bisect_right(train_ends, hi_str)
    candidates = all_base_periods[left:right]

    periods = [
        generate_full_periods_train_test_valid(
            ts, te,
            test_period_months=test_months,
            validation_period_months=config.experiment.validation_period_months,
            validation_offset_months=config.experiment.validation_offset_months,
            test_period_start_offset_months=offset,
            prediction_horizon=horizon,
            trading_calendar=config.experiment.trading_calendar,
            trading_dates=trading_dates,
        )
        for (ts, te) in candidates
    ]

    return [p for p in periods if p[-1] >= train_start and p[-2] <= train_end]
        