import streamlit as st
import pandas as pd
import numpy as np

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
    unit_profit_growth_rate = params['profit_growth_rate']  # Monthly rate
    fixed_cost_growth_rate = params['fixed_cost_growth_rate']  # Monthly rate
    fixed_cost_cap = params['fixed_cost_cap']
    diseconomies_of_scale_rate = params['diseconomies_of_scale']  # Rate (e.g., 0.005 for 0.5%)
    months = params['months']
    ci_months = params['ci_months']
    ci_amounts = params['ci_amounts']

    # Create capital injection dictionary for quick lookup
    capital_injections = dict(zip(ci_months, ci_amounts))

    # Initialize variables for the loop
    starting_capital = initial_capital
    current_fixed_costs = fixed_costs_start
    monthly_data_list = []

    for month in range(1, months + 1):
        # Capital available after fixed costs
        capital_for_production = max(0, starting_capital - current_fixed_costs)

        # Units produced
        units_produced = 0
        if variable_cost_per_unit > 0:
            units_produced = int(capital_for_production // variable_cost_per_unit)

        # Calculate costs and revenue for the month
        monthly_variable_costs = units_produced * variable_cost_per_unit
        revenue = units_produced * selling_price_per_unit_start
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
            'Var Cost/Unit': variable_cost_per_unit,
            'Sell Price/Unit': selling_price_per_unit_start
        })

        # Update fixed costs for next month (with cap)
        current_fixed_costs = min(current_fixed_costs * (1 + fixed_cost_growth_rate), fixed_cost_cap)

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
    yearly_agg = pd.DataFrame()  # Initialize empty
    if df['Year'].max() > 0:
        yearly_agg = df.groupby('Year').agg(agg_rules)

    # --- Calculate Financial Summary Metrics (Quarterly & Yearly) ---
    def calculate_summary(agg_df):
        summary = agg_df.copy()
        summary['Profit Margin (%)'] = (summary['Profit'] / summary['Revenue'].replace(0, np.nan)) * 100
        summary['Total Assets (Est.)'] = summary['Ending Capital']  # Approximation
        summary['Total Liabilities (Est.)'] = 0  # Approximation
        summary['Equity (Est.)'] = summary['Ending Capital']  # Approximation
        return summary

    quarterly_summary = calculate_summary(quarterly_agg)
    yearly_summary = calculate_summary(yearly_agg) if not yearly_agg.empty else pd.DataFrame()

    # --- Calculate KPIs (Quarterly & Yearly) ---
    def calculate_kpis(agg_df):
        kpi = pd.DataFrame(index=agg_df.index)
        # Growth Rates (handle division by zero and missing previous period)
        kpi['Revenue Growth Rate (%)'] = agg_df['Revenue'].pct_change() * 100
        kpi['Profit Growth Rate (%)'] = agg_df['Profit'].pct_change() * 100
        # Margins
        kpi['Gross Profit Margin (%)'] = ((agg_df['Revenue'] - agg_df['Variable Costs']) / agg_df['Revenue'].replace(0, np.nan)) * 100
        kpi['Operating Profit Margin (%)'] = ((agg_df['Revenue'] - agg_df['Total Costs']) / agg_df['Revenue'].replace(0, np.nan)) * 100
        kpi['Net Profit Margin (%)'] = (agg_df['Profit'] / agg_df['Revenue'].replace(0, np.nan)) * 100
        # Units & Per Unit
        kpi['Units Produced'] = agg_df['Units Produced']
        kpi['Revenue per Unit'] = agg_df['Revenue'] / agg_df['Units Produced'].replace(0, np.nan)
        kpi['Cost per Unit'] = agg_df['Total Costs'] / agg_df['Units Produced'].replace(0, np.nan)
        return kpi

    quarterly_kpis = calculate_kpis(quarterly_agg)
    yearly_kpis = calculate_kpis(yearly_agg) if not yearly_agg.empty else pd.DataFrame()

    return quarterly_summary, yearly_summary, quarterly_kpis, yearly_kpis


# --- Table Creation ---
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
                    row_data[q_label] = np.nan  # Metric not found in this dataset
            # Yearly values
            if y_cols:
                for y_label, y_series in y_cols.items():
                    if data_key in y_series:
                        row_data[y_label] = y_series[data_key]
                    else:
                        row_data[y_label] = np.nan  # Metric not found
            else:
                # Add placeholder columns if no yearly data exists but expected
                if any(k.startswith('Y') for k in list(q_cols.keys()) + list(y_cols.keys())):  # Check if yearly columns expected
                    num_years_expected = len(q_data) // 4  # Estimate
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
        result_df = pd.DataFrame(combined_data).T  # Transpose here!
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
        formatted_df = df_to_format.copy().astype(object)  # Work on copy, ensure object type
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
                else:  # Assume currency
                    formatted_df.loc[metric, col_name] = format_currency(value, currency)
        return formatted_df

    return format_df(fin_table), format_df(kpi_table)


# --- Chart Creation (using Streamlit's native charts) ---
def create_charts(monthly_df, q_summary, q_kpis, currency):
    charts = {}
    divisor = INR_TO_USD if currency == 'USD' else 1.0

    # 1. Monthly Trends Chart
    if not monthly_df.empty:
        chart_df_monthly = monthly_df[['Month', 'Revenue', 'Total Costs', 'Profit', 'Ending Capital']].copy()
        for col in ['Revenue', 'Total Costs', 'Profit', 'Ending Capital']:
            chart_df_monthly[col] /= divisor
        chart_df_monthly = chart_df_monthly.set_index('Month')
        charts['monthly'] = chart_df_monthly

    # 2. Quarterly Financial Summary Chart
    if not q_summary.empty:
        chart_df_q_summary = q_summary[['Revenue', 'Profit', 'Ending Capital']].copy()
        for col in ['Revenue', 'Profit', 'Ending Capital']:
            chart_df_q_summary[col] /= divisor
        chart_df_q_summary.index.name = 'Quarter'
        charts['q_summary'] = chart_df_q_summary

    # 3. Quarterly KPIs Chart (Percentages)
    if not q_kpis.empty:
        # Select only percentage KPIs that are usually plotted together
        kpi_cols_to_plot = ['Revenue Growth Rate (%)', 'Profit Growth Rate (%)', 'Gross Profit Margin (%)', 'Net Profit Margin (%)']
        # Filter out KPIs not present or completely NaN
        valid_kpi_cols = [col for col in kpi_cols_to_plot if col in q_kpis.columns and not q_kpis[col].isnull().all()]
        if valid_kpi_cols:
            chart_df_q_kpis = q_kpis[valid_kpi_cols].copy()
            chart_df_q_kpis.index.name = 'Quarter'
            charts['q_kpis'] = chart_df_q_kpis
        else:
            charts['q_kpis'] = pd.DataFrame()  # Ensure empty df if no valid kpis

    return charts


# --- Streamlit UI ---
st.title("📊 Financial Projection Tool")

# --- Inputs ---
with st.expander("Configuration Inputs", expanded=True):
    st.sidebar.header("Currency")
    currency_choice = st.sidebar.radio("Select Currency", ('INR (₹)', 'USD ($)'), key='currency')
    selected_currency = 'USD' if currency_choice == 'USD ($)' else 'INR'

    # Use a form to gather inputs and run calculations on submission
    with st.form(key='projection_form'):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Core Inputs")
            initial_capital = st.number_input("Initial Capital", min_value=0.0, value=200000.0, step=10000.0, format="%.0f")
            fixed_costs = st.number_input("Initial Fixed Costs / Month", min_value=0.0, value=30000.0, step=1000.0, format="%.0f")
            variable_cost_per_unit = st.number_input("Variable Cost Per Unit", min_value=0.0, value=13.0, step=0.5, format="%.2f")
            selling_price_per_unit = st.number_input("Selling Price Per Unit", min_value=0.0, value=20.0, step=0.5, format="%.2f")
            months = st.number_input("Number of Months", min_value=1, value=24, step=1)

        with col2:
            st.subheader("Growth & Scaling")
            profit_growth_rate_pct = st.number_input("Unit Profit Growth Rate (% / Month)", value=0.0, step=0.1, format="%.1f")
            fixed_cost_growth_rate_pct = st.number_input("Fixed Cost Growth Rate (% / Month)", value=1.0, step=0.1, format="%.1f")
            fixed_cost_cap = st.number_input("Fixed Cost Cap / Month", min_value=0.0, value=1000000.0, step=10000.0, format="%.0f")
            diseconomies_scale_pct = st.number_input("Diseconomies of Scale (% of Revenue)", min_value=0.0, value=0.5, step=0.1, format="%.2f")

        st.subheader("Capital Injections (Optional)")
        col3, col4 = st.columns(2)
        with col3:
            ci_months_str = st.text_input("Injection Months (comma-separated)", value="", placeholder="e.g., 6, 12, 18")
        with col4:
            ci_amounts_str = st.text_input("Injection Amounts (comma-separated)", value="", placeholder="e.g., 50000, 100000, 150000")

        # Submit button for the form
        submitted = st.form_submit_button("Generate Projections")

# --- Processing and Display ---
if submitted:
    # Process inputs only when form is submitted
    try:
        ci_months = [int(m.strip()) for m in ci_months_str.split(',') if m.strip()] if ci_months_str else []
        ci_amounts = [float(a.strip()) for a in ci_amounts_str.split(',') if a.strip()] if ci_amounts_str else []
        if len(ci_months) != len(ci_amounts):
            st.error("Number of capital injection months must match number of amounts.")
            st.stop()  # Stop execution if lists don't match
    except ValueError:
        st.error("Please enter comma-separated numbers for capital injection months and amounts.")
        st.stop()

    input_params = {
        'initial_capital': initial_capital,
        'fixed_costs': fixed_costs,
        'variable_cost_per_unit': variable_cost_per_unit,
        'selling_price_per_unit': selling_price_per_unit,
        'profit_growth_rate': profit_growth_rate_pct / 100.0,
        'fixed_cost_growth_rate': fixed_cost_growth_rate_pct / 100.0,
        'fixed_cost_cap': fixed_cost_cap,
        'diseconomies_of_scale': diseconomies_scale_pct / 100.0,
        'months': months,
        'ci_months': ci_months,
        'ci_amounts': ci_amounts
    }

    # --- Run Simulation & Aggregation ---
    with st.spinner('Calculating projections...'):
        monthly_results_df = run_simulation(input_params)
        q_summary_raw, y_summary_raw, q_kpis_raw, y_kpis_raw = aggregate_and_calculate_kpis(monthly_results_df)

        # Store results in session state to survive currency changes without recalculating
        st.session_state['monthly_results'] = monthly_results_df
        st.session_state['q_summary_raw'] = q_summary_raw
        st.session_state['y_summary_raw'] = y_summary_raw
        st.session_state['q_kpis_raw'] = q_kpis_raw
        st.session_state['y_kpis_raw'] = y_kpis_raw
        st.session_state['results_generated'] = True

# --- Display Results (if generated) ---
if st.session_state.get('results_generated', False):
    # Retrieve data from session state
    monthly_df = st.session_state['monthly_results']
    q_summary_raw = st.session_state['q_summary_raw']
    y_summary_raw = st.session_state['y_summary_raw']
    q_kpis_raw = st.session_state['q_kpis_raw']
    y_kpis_raw = st.session_state['y_kpis_raw']

    # Apply formatting based on current currency choice
    fin_summary_table_formatted, kpi_table_formatted = create_summary_table(
        q_summary_raw, y_summary_raw, q_kpis_raw, y_kpis_raw, selected_currency
    )

    # Format monthly table
    monthly_table_formatted = monthly_df.copy()
    currency_cols = ['Starting Capital', 'Fixed Costs', 'Variable Costs', 'Diseconomies Cost', 'Total Costs', 'Revenue', 'Profit', 'Capital Injection', 'Ending Capital']
    for col in currency_cols:
        monthly_table_formatted[col] = monthly_table_formatted[col].apply(lambda x: format_currency(x, selected_currency))

    # Display Tables
    st.subheader("Monthly Results")
    st.dataframe(monthly_table_formatted)

    st.subheader("Financial Summary")
    st.dataframe(fin_summary_table_formatted)

    st.subheader("KPIs")
    st.dataframe(kpi_table_formatted)

    # Charts
    charts = create_charts(monthly_df, q_summary_raw, q_kpis_raw, selected_currency)

    st.subheader("Monthly Trends")
    if 'monthly' in charts and not charts['monthly'].empty:
        st.line_chart(charts['monthly'])

    st.subheader("Quarterly Financial Summary")
    if 'q_summary' in charts and not charts['q_summary'].empty:
        st.bar_chart(charts['q_summary'])

    st.subheader("Quarterly KPIs")
    if 'q_kpis' in charts and not charts['q_kpis'].empty:
        st.line_chart(charts['q_kpis'])
