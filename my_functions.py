# Author: Pimentel, Esteban A
# Contact: eap.pimentel@gmail.com
# Publish Date: 2023-03-23
# Customer: Sheikh, Jim


import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import pdb
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interactive
import json


########################################################################
###############   CORE PHYSICS   ###############
########################################################################

def run_scenario(purchase_price, scenario):
    """
    Compute key financials which are can be utilized to calc performance metrics
        - price <double>: purchase price of rental property
        - scenario <dict>: specifies property characteristics, financing, and environment
    Returns:
        - results <dict>: contains key values and series
    """

    # utility
    horizon = scenario['horizon_yrs']
    periods_per_year = scenario['periods_per_year']
    n_periods = horizon * periods_per_year
    purchase_price = np.array(purchase_price)

    # get values at each period from borrowed funds
    interest_charges, principal_pmt, principal_i = run_financing(purchase_price, scenario['financing'], horizon, periods_per_year)

    # find inflation figures at each period
    cpi_period = np.power(1 + scenario['cpi_yoy'][0:n_periods], 1/periods_per_year) - 1
    cpi_cum = (1+cpi_period).cumprod()

    ppi_period = np.power(1 + scenario['ppi_yoy'][0:n_periods], 1/periods_per_year) - 1
    ppi_cum = (1+ppi_period).cumprod()


    # get rental cpi: updates each Q1 with inflation from each of the past 4 quarters
    # make each row a year through reshape
    cpi_rental_cum = np.full(len(cpi_cum), 1.0)
    if scenario['is_inflation_indexed']:
        year_end_cpi = cpi_cum.reshape((-1,periods_per_year))[:,-1][0:-1]
        cpi_rental_cum = np.insert(year_end_cpi,0,1)
        cpi_rental_cum = np.repeat(cpi_rental_cum, 4)

    # calc income and costs from rental unit
    rental_income = scenario['gross_income']*(1-scenario['vacancy_rate'])*cpi_rental_cum / periods_per_year
    maintenance_cost = (scenario['gross_income'] - scenario['net_income'])*ppi_cum / periods_per_year

    # track value of rental property, assume linear, recognize appreciation as profit each period
    net_income_at_exit = (rental_income[-1] - maintenance_cost[-1]) * periods_per_year
    exit_price =  net_income_at_exit / scenario['rrr_exit'] * (1 + scenario['deflated_cap_gain'])
    appreciation = (exit_price - purchase_price) / n_periods

    # organize results into dictionary
    results = {
        'interest_charges': interest_charges
        , 'principal_pmt': principal_pmt
        , 'principal_i': principal_i
        , 'cpi_cum': cpi_cum
        , 'rental_income': rental_income
        , 'maintenance_cost': maintenance_cost
        , 'exit_price': exit_price
    }

    return results


def run_financing(purchase_price, terms, horizon, periods_per_year):
    """
    Calculate the interest charge and starting principal at each time period.
    Uses simple interest, so the interest rate used in a period is APR/periods_per_year

    Returns:
    - interest_charges <np.array>: interest charged at end of each period
    - principal_pmt <np.array>: amount used to pay down principal at end of each period
    - principal_i <np.array>: remaining loan balance at the beginning of each period
    """

    n_periods = horizon * periods_per_year

    # get interest rate to use at each period
    if terms['is_fixed_rate']:
        interest_rate = np.full(shape=n_periods, fill_value= terms['fixed_rate'])
    else:
        interest_rate = terms['index_rate'][0:n_periods] + terms['margin']
    # adjust to simple interest
    interest_rate = interest_rate / periods_per_year


    ## find principal at the beginning of each period
    if terms['is_amortizing']:

        principal = list()
        principal.append( terms['ltv_ratio'] * purchase_price )
        principal_pmt = list()

        tot_periods = terms['term']*periods_per_year

        for i in range(n_periods):
            pmt = get_payment(principal[i], interest_rate[i], tot_periods - i)
            interest = principal[i]*interest_rate[i]
            principal_pmt.append(pmt - interest)
            # get starting principal for next time step
            principal.append( principal[i] - principal_pmt[i])
        principal_i = np.array(principal[0:n_periods])
        principal_pmt = np.array(principal_pmt)

    # if not amoritzing
    else:
        principal_i = np.full(shape= n_periods, fill_value= terms['ltv_ratio']*purchase_price)
        principal_pmt = np.full(shape= n_periods, fill_value=0)

    ## calc interest charges
    interest_charges = principal_i * interest_rate

    ## calc princip

    return interest_charges, principal_pmt, principal_i


def get_payment(principal, rate, term):
    """
    Calculate the payment.
    """
    pmt = principal * rate * (1+rate)**term / ((1+rate)**term - 1)
    return pmt


########################################################################
###############   PERFORMANCE METRIC : IRR   ###############
########################################################################

def get_irr(purchase_price, scenario):
    """
    Calculate the IRR for this scenario by using goal seek on inflation-adjusted cashflows
    """

    real_cashflow = get_real_cashflow(purchase_price, scenario)

    obj = lambda x: calc_npv(real_cashflow, x, scenario)**2
    # irr = minimize_scalar(obj, bounds=(-0.5,0.5), tol=1e-4)['x']
    irr = minimize_scalar(obj)['x']
    return round(irr,4)


def get_real_cashflow(purchase_price, scenario):
    """
    Run the scenario and calculate the inflation-adjusted cashflow
    """

    # utility
    horizon = scenario['horizon_yrs']
    periods_per_year = scenario['periods_per_year']
    n_periods = horizon * periods_per_year
    purchase_price = np.array(purchase_price)

    # run physics
    res = run_scenario(purchase_price, scenario)

    # calc cashflow
    final_loan_balance = res['principal_i'][-1] - res['principal_pmt'][-1]
    down_payment = purchase_price - res['principal_i'][0]

    cashflow = res['rental_income'] - res['maintenance_cost'] - res['principal_pmt'] - res['interest_charges']
    cashflow[-1] += res['exit_price'] - final_loan_balance

    real_cashflow = cashflow / res['cpi_cum']
    real_cashflow = np.insert(real_cashflow, 0, -1*down_payment)  # add the year 0 entry

    return real_cashflow


def calc_npv(real_cashflow, discount_rate, scenario):
    """
    Calculate NPV given a cashflow and discount rate
    """

    # utility
    horizon = scenario['horizon_yrs']
    periods_per_year = scenario['periods_per_year']
    n_periods = horizon * periods_per_year

    # ignore overflow warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        discount_period = np.sign(1+discount_rate) * np.abs(1 + discount_rate) ** (1/periods_per_year)
        discount_cum = np.power(discount_period, np.arange(n_periods+1))

    dcf = real_cashflow / discount_cum
    npv = dcf.sum()

    return npv


########################################################################
###############   MACRO ENVIRONMENT MODEL  ###############
########################################################################

def sample_single_environment(params):

    # translate inputs
    reversion_factor = 1/(params['avg_yrs_to_revert']*4)
    var_names = ['cpi','ppi','ir']

    start_vals = np.array([params[v]['start_val'] for v in var_names])
    floor_vals = np.array([params[v]['floor'] for v in var_names])
    long_run_vals = np.array([params[v]['long_run'] for v in var_names])
    std = np.array([params[v]['std'] for v in var_names])

    # build covariance matrix
    cov = np.zeros(shape=(3,3))
    for i in range(3):
        cov[i,i] = std[i]**2
        for j in range(i):
            cov[i,j] = params['corr'] * std[i] * std[j]
            cov[j,i] = cov[i,j]

    # sample variances for drifts, and likelihood of tail events
    sample = np.random.multivariate_normal(np.zeros(3), cov, params['n_periods'])
    is_tail_event = np.random.uniform(size = params['n_periods']) < params['p_tail']

    sim_vals = np.zeros(shape= (params['n_periods']+1, 3))
    sim_vals[0,:] = start_vals.copy()
    for i in range(1,params['n_periods']+1):
        cur_val = sim_vals[i-1,:]
        if is_tail_event[i-1]:
            sim_vals[i,:] = cur_val + sample[i-1,:]*params['tail_std_multiple'] + params['tail_event_mean']
        else:
            sim_vals[i,:] = cur_val + (long_run_vals - cur_val)*reversion_factor + sample[i-1,:]

        # limit how negative values can go
        sim_vals[i,:] = np.maximum(sim_vals[i,:], floor_vals)

    # aggregate data into df
    data = { tag:sim_vals[:,i] for i, tag in enumerate(['cpi','ppi','ir'])}
    data['period'] = range(0,params['n_periods']+1)
    data['is_tail_event'] = np.insert(is_tail_event, 0, False)
    df = pd.DataFrame(data)

    return df

def mass_sample_environments(params):
    N = params['n_samples']
    df_list = list()
    np.random.seed(params['random_seed'])
    for i in range(N):
        df = sample_single_environment(params)
        df['sample_id'] = i
        df_list.append(df)

    df_full = pd.concat(df_list)
    df_full.reset_index(inplace=True, drop=True)

    col_order = ['sample_id', 'period', 'cpi', 'ppi', 'ir', 'is_tail_event']
    df_full = df_full[col_order]

    return df_full


########################################################################
###############   GOAL SEEK for PURCHASE PRICE   ###############
########################################################################

def build_objective(scenario, performance_func, target):
    """
    Returns a function that calculate the error between ROE for a purchase price and the target_ROE
        - min_roe : minimum acceptable ROE for the investment
    """

    def objective(x):
        """
        Calculates error from min_roe
        """
        res = performance_func(x, scenario)
        err = (target - res)**2
        return err
    return objective


def find_best_offer(scenario, performance_func, target):
    """
    Uses optimization to backsolve for purchase price that yields target performance metric
    """
    obj = build_objective(scenario, performance_func, target)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # result = minimize_scalar(obj
        #             , bounds = (scenario['gross_income'], scenario['gross_income']*1e3)
        #             , tol = 1e-8
        #         )['x']
        result = minimize_scalar(obj)['x']

    return round(result)

def run_all_scenarios(df_full, scenario, performance_func, target):
    n_samples = df_full['sample_id'].max()

    purchase_price_list = list()
    for i in range(n_samples):
        tot_periods = scenario['horizon_yrs']*scenario['periods_per_year']
        # get sampled env
        id_mask = df_full['sample_id'] == i
        period_mask = np.logical_and(df_full['period'] > 0, df_full['period'] <= tot_periods)
        mask = np.logical_and(id_mask, period_mask)
        subset = df_full[mask]
        # replace in scenario
        scenario['cpi_yoy'] = subset['cpi'].to_numpy()
        scenario['ppi_yoy'] = subset['ppi'].to_numpy()
        scenario['financing']['index_rate'] = subset['ir'].to_numpy()
        # run
        purchase_price_list.append( find_best_offer(scenario, performance_func, target) )

    data = {'sample_id':range(n_samples), 'purchase_price': purchase_price_list}
    df = pd.DataFrame(data)

    return df


################################################################################################
################## FIND PERFORMANCE given FIXED PURCHASE PRICE  ########################
################################################################################################

def run_all_get_performance(df_full, scenario, purchase_price, performance_func, metric_name):
    n_samples = df_full['sample_id'].max()

    performance_list = list()
    for i in range(n_samples):
        tot_periods = scenario['horizon_yrs']*scenario['periods_per_year']
        # get sampled env
        id_mask = df_full['sample_id'] == i
        period_mask = np.logical_and(df_full['period'] > 0, df_full['period'] <= tot_periods)
        mask = np.logical_and(id_mask, period_mask)
        subset = df_full[mask]
        # replace in scenario
        scenario['cpi_yoy'] = subset['cpi'].to_numpy()
        scenario['ppi_yoy'] = subset['ppi'].to_numpy()
        scenario['financing']['index_rate'] = subset['ir'].to_numpy()
        # run
        performance_list.append( performance_func(purchase_price, scenario) )

    data = {'sample_id':range(n_samples), metric_name: performance_list}
    df = pd.DataFrame(data)

    return df


################################################
################## PLOTTING  ########################
################################################

def plot_single_environment(df_full, sample_id, ylims=(0,0.15)):

    df_filter = df_full[df_full.sample_id == sample_id]
    df_melt = pd.melt(df_filter, id_vars=['period', 'is_tail_event'], value_vars = ['cpi','ppi','ir'], var_name = "var", value_name="rate")


    fig, ax = plt.subplots()
    sns.lineplot(data=df_melt, y = "rate", x = 'period', hue="var", ax= ax)
    ax.set_ylim(ylims)
    plt.show()

def plot_all_scenarios(df_full, var, ylims=(0, 0.2), alpha=0.3):
    palette = sns.color_palette(['black'], len(df_full['sample_id'].unique()))

    fig, ax = plt.subplots()
    p = sns.lineplot(data=df_full, y=var, x='period', hue='sample_id', palette= palette, legend = False, alpha = alpha, ax = ax, linewidth=.5)
    sns.scatterplot(data=df_full[df_full['is_tail_event'] == True], x = 'period', y=var, s=5)

    ax.set_title(f'{str.upper(var)} series samples')
    ax.set_ylabel(str.upper(var))
    ax.set_ylim(ylims)


################################################
################## LOGGING ##################
################################################

def write_run_specs(scenario, dist_params, target_irr, purchase_price, run_tag):
    # remove some vals that are filled in accordng to sample_id
    scenario.pop('cpi_yoy', None)
    scenario.pop('ppi_yoy', None)
    scenario['financing'].pop('index_rate', None)
    # assemble full_specs
    full_specs = {
        'target_irr': target_irr
        , 'fixed_purchase_price': purchase_price
        , 'environment': dist_params
        , 'model': scenario
    }
    # write file
    with open(f"specs_{run_tag}.json", "w") as fp:
        json.dump(full_specs,fp)


################################################
################## DEPRECATED ##################
################################################
def get_roe(purchase_price, scenario):
    """
    Calculate the weighted average ROE for this scenario
    """

    # utility
    horizon = scenario['horizon_yrs']
    periods_per_year = scenario['periods_per_year']
    n_periods = horizon * periods_per_year
    purchase_price = np.array(purchase_price)

    # run physics
    res = run_scenario(purchase_price, scenario)

    # calc performance metric
    appreciation = (res['exit_price'] - purchase_price) / n_periods  #linear appreciation
    profit = res['rental_income'] + appreciation - res['interest_charges'] - res['maintenance_cost']
    real_profit = profit / res['cpi_cum']

    # get equity at the beginning of each period, adjust for inflation
    asset_val_i = np.linspace(start= purchase_price, stop= res['exit_price'], num= n_periods, endpoint=False)
    equity_i = asset_val_i - res['principal_i']
    cpi_i_cum = np.insert(res['cpi_cum'], 0, 1)[0:n_periods]  #cpi_cum at beginning of period
    real_equity = equity_i / cpi_i_cum

    # get ROE
    roe_real_annual = real_profit.sum() / real_equity.sum() * periods_per_year
    # note: this definition is the weighted average of ROE in each period,
    # accounts for change in equity in the deal over lifetime

    return roe_real_annual
