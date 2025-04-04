import streamlit as st
import pandas as pd
import numpy as np
import json

# --- Constants & Configuration ---
st.set_page_config(layout="wide", page_title="Financial Projection Tool")
INR_TO_USD = 85.0

# --- Formatting Helpers ---
def format_currency(amount, currency, use_units=True):
    if pd.isna(amount) or not np.isfinite(amount):
        return "N/A"

    value = amount / (INR_TO_USD if currency == 'USD' else 1)
    sign = '-' if value < 0 else ''
    absValue = abs(value)

    if not use_units:
        currency_symbol = "$" if currency == 'USD' else "₹"
        return f"{currency_symbol}{value:,.2f}"

    if absValue >= 1e9:
        return f"{sign}{(absValue / 1e9):.1f} bn"
    elif absValue >= 1e6:
        return f"{sign}{(absValue / 1e6):.1f} mn"
    elif absValue >= 1e3:
        return f"{sign}{(absValue / 1e3):.1f} k"
    else:
        return f"{sign}{value:.0f}"

def format_units(units):
    if pd.isna(units) or not np.isfinite(units):
        return "N/A"
    if abs(units) >= 1e9:
        return f"{(units / 1e9):.1f} bn"
    elif abs(units) >= 1e6:
        return f"{(units / 1e6):.1f} mn"
    elif abs(units) >= 1e3:
        return f"{(units / 1e3):.1f} k"
    else:
        return f"{units:.0f}"

def format_percentage(value):
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:.1f}%"

# --- Core Calculation Logic ---
def run_simulation(params):
    # Destructure parameters
    initial_capital = params['initial_capital']
    fixed_costs_start = params['fixed_costs']
    variable_cost_per_unit = params['variable_cost_per_unit']
    selling_price_per_unit_start = params['selling_price_per_unit']
    unit_profit_growth_rate = params['profit_growth_rate'] # Monthly rate
    fixed_cost_growth_rate = params['fixed_cost_growth_rate'] # Monthly rate
    fixed_cost_cap = params['fixed_cost_cap']
    diseconomies_of_scale_rate = params['diseconomies_of_scale'] # Rate (e.g., 0.005 for 0.5%)
    months = params['months']
    ci_months = params['ci_months']
    ci_amounts = params['ci_amounts']

    # Create capital injection dictionary for quick lookup
    capital_injections = dict(zip(ci_months, ci_amounts))

    # Initialize variables for the loop
    starting_capital = initial_capital
    current_fixed_costs = fixed_costs_start
    current_variable_cost_per_unit = variable_cost_per_unit
    current_selling_price_per_unit = selling_price_per_unit_start

    monthly_data_list = []

    for month in range(1, months + 1):
        # Capital available after fixed costs
        capital_for_production = max(0, starting_capital - current_fixed_costs)

        # Units produced
        units_produced = 0
        if current_variable_cost_per_unit > 0:
             units_produced = int(capital_for_production // current_variable_cost_per_unit)

        # Calculate costs and revenue for the month
        monthly_variable_costs = units_produced * current_variable_cost_per_unit
        revenue = units_produced * current_selling_price_per_unit
        diseconomies_cost = revenue * diseconomies_of_scale_rate
        total_costs = current_fixed_costs + monthly_variable_costs + diseconomies_cost

        # Profit
        monthly_profit = revenue - total_costs

        # Capital Injection for this month
        capital_injection = capital_injections.get(month, 0)

        # Ending Capital
        ending_capital = starting_capital + monthly_profit + capital_injection

        # Store results
        monthly_data_list.append({
            'Month': month,
            'Starting Capital': starting_capital,
            'Fixed Costs': current_fixed_costs,
            'Units Produced': units_produced,
            'Variable Costs': monthly_variable_costs,
            'Diseconomies Cost': diseconomies_cost,
            'Total Costs': total_costs,
            'Revenue': revenue,
            'Profit': monthly_profit,
            'Capital Injection': capital_injection,
            'Ending Capital': ending_capital,
            # Store unit economics for potential later use (like KPI calc)
            'Var Cost/Unit': current_variable_cost_per_unit,
            'Sell Price/Unit': current_selling_price_per_unit
        })

        # --- Update values for the *next* month ---
        starting_capital = ending_capital

        # Grow fixed costs, capped
        current_fixed_costs = min(current_fixed_costs * (1 + fixed_cost_growth_rate), fixed_cost_cap)

        # Grow profit per unit by adjusting selling price
        current_profit_per_unit = current_selling_price_per_unit - current_variable_cost_per_unit
        next_profit_per_unit = current_profit_per_unit * (1 + unit_profit_growth_rate)
        current_selling_price_per_unit = current_variable_cost_per_unit + next_profit_per_unit
        # We assume variable cost per unit is constant unless explicitly changed by another input

    return pd.DataFrame(monthly_data_list)

# --- Data Aggregation and KPI Calculation ---
def aggregate_and_calculate_kpis(monthly_df):
    if monthly_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = monthly_df.copy()
    df['Quarter'] = (df['Month'] - 1) // 3 + 1
    df['Year'] = (df['Month'] - 1) // 12 + 1

    # Define aggregation rules
    agg_rules = {
        'Revenue': 'sum',
        'Fixed Costs': 'sum',
        'Variable Costs': 'sum',
        'Diseconomies Cost': 'sum',
        'Total Costs': 'sum',
        'Profit': 'sum',
        'Units Produced': 'sum',
        'Capital Injection': 'sum',
        'Starting Capital': 'first',
        'Ending Capital': 'last'
    }

    # Aggregate quarterly
    quarterly_agg = df.groupby('Quarter').agg(agg_rules)

    # Aggregate yearly (if applicable)
    yearly_agg = pd.DataFrame() # Initialize empty
    if df['Year'].max() > 0:
        yearly_agg = df.groupby('Year').agg(agg_rules)

    # --- Calculate Financial Summary Metrics (Quarterly & Yearly) ---
def create_summary_table(q_summary, y_summary, q_kpis, y_kpis, currency):
    # Define which metrics go where
    fin_summary_metrics = {
        'Revenue': 'Revenue', 'Operating Expenses': 'Total Costs', 'Net Profit': 'Profit',
        'Profit Margin (%)': 'Profit Margin (%)', 'Starting Capital': 'Starting Capital',
        'Ending Capital': 'Ending Capital', 'Total Assets (Est.)': 'Total Assets (Est.)',
        'Total Liabilities (Est.)': 'Total Liabilities (Est.)', 'Equity (Est.)': 'Equity (Est.)'
    }
    kpi_metrics = {
        'Revenue Growth Rate (%)': 'Revenue Growth Rate (%)', 'Profit Growth Rate (%)': 'Profit Growth Rate (%)',
        'Gross Profit Margin (%)': 'Gross Profit Margin (%)', 'Operating Profit Margin (%)': 'Operating Profit Margin (%)',
        'Net Profit Margin (%)': 'Net Profit Margin (%)', 'Units Produced': 'Units Produced',
        'Revenue per Unit': 'Revenue per Unit', 'Cost per Unit': 'Cost per Unit'
    }

    # Function to build one transposed table
def build_table(metrics_map, q_data, y_data):
        q_cols = {f"Q{i}": q_data.loc[i] for i in q_data.index}
        y_cols = {}
        if not y_data.empty:
            y_cols = {f"Y{i} Total": y_data.loc[i] for i in y_data.index}

        combined_data = {}
        for display_name, data_key in metrics_map.items():
            row_data = {}
            # Quarterly values
            for q_label, q_series in q_cols.items():
                if data_key in q_series:
                    row_data[q_label] = q_series[data_key]
                else:
                    row_data[q_label] = np.nan # Metric not found in this dataset

            # Yearly values
            if y_cols:
                for y_label, y_series in y_cols.items():
                    if data_key in y_series:
                       row_data[y_label] = y_series[data_key]
                    else:
                       row_data[y_label] = np.nan # Metric not found
            else:
                 # Add placeholder columns if no yearly data exists but expected
                 if any(k.startswith('Y') for k in list(q_cols.keys()) + list(y_cols.keys())): # Check if yearly columns expected
                     num_years_expected = len(q_data)//4 # Estimate
                     for i in range(1, num_years_expected + 1):
                          row_data[f'Y{i} Total'] = np.nan

            # Q/Q and Y/Y Changes (use latest available period's calculated growth/value)
            last_q_key = f"Q{q_data.index.max()}"
            if last_q_key in row_data and len(q_data) > 1:
                 # For growth rates/margins, Q/Q change is just the latest value
                 if '%' in display_name:
                     row_data['Q/Q Change'] = q_data.loc[q_data.index.max(), data_key] if data_key in q_data.columns else np.nan
                 # For absolute values, calculate % change if possible
                 elif q_data.index.max() > q_data.index.min():
                     prev_q_val = q_data.loc[q_data.index.max() - 1, data_key] if data_key in q_data.columns else np.nan
                     last_q_val = q_data.loc[q_data.index.max(), data_key] if data_key in q_data.columns else np.nan
                     if not pd.isna(prev_q_val) and not pd.isna(last_q_val) and prev_q_val != 0:
                         row_data['Q/Q Change'] = ((last_q_val / prev_q_val) - 1) * 100
                     else:
                         row_data['Q/Q Change'] = np.nan
                 else:
                      row_data['Q/Q Change'] = np.nan
            else:
                row_data['Q/Q Change'] = np.nan

            if not y_data.empty and y_data.index.max() > y_data.index.min():
                 last_y_key = f"Y{y_data.index.max()} Total"
                 # Similar logic for Y/Y change
                 if '%' in display_name:
                     row_data['Y/Y Change'] = y_data.loc[y_data.index.max(), data_key] if data_key in y_data.columns else np.nan
                 elif y_data.index.max() > y_data.index.min():
                     prev_y_val = y_data.loc[y_data.index.max() - 1, data_key] if data_key in y_data.columns else np.nan
                     last_y_val = y_data.loc[y_data.index.max(), data_key] if data_key in y_data.columns else np.nan
                     if not pd.isna(prev_y_val) and not pd.isna(last_y_val) and prev_y_val != 0:
                         row_data['Y/Y Change'] = ((last_y_val / prev_y_val) - 1) * 100
                     else:
                          row_data['Y/Y Change'] = np.nan
                 else:
                     row_data['Y/Y Change'] = np.nan
            else:
                 row_data['Y/Y Change'] = np.nan

            combined_data[display_name] = row_data

        # Create DataFrame and order columns
        result_df = pd.DataFrame(combined_data).T # Transpose here!
        col_order = list(q_cols.keys()) + list(y_cols.keys()) + ['Q/Q Change', 'Y/Y Change']
        # Ensure all expected columns exist, adding NaN columns if needed
        for col in col_order:
            if col not in result_df.columns:
                result_df[col] = np.nan
        return result_df[col_order]

    # Build the two tables
    fin_table = build_table(fin_summary_metrics, q_summary, y_summary)
    kpi_table = build_table(kpi_metrics, q_kpis, y_kpis)

    # Apply Formatting
def format_df(df_to_format):
        formatted_df = df_to_format.copy().astype(object) # Work on copy, ensure object type
        for metric, row in df_to_format.iterrows():
            for col_name, value in row.items():
                if pd.isna(value):
                   formatted_df.loc[metric, col_name] = 'N/A'
                   continue

                if '%' in metric or 'Change' in col_name:
                    formatted_df.loc[metric, col_name] = format_percentage(value)
                elif 'Unit' in metric:
                     formatted_df.loc[metric, col_name] = format_currency(value, currency, use_units=False)
                elif 'Units' in metric:
                    formatted_df.loc[metric, col_name] = format_units(value)
                else: # Assume currency
                    formatted_df.loc[metric, col_name] = format_currency(value, currency)
        return formatted_df

return format_df(fin_table), format_df(kpi_table)
