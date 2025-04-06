import streamlit as st
import pandas as pd
import plotly.express as px
import json
import math

# --- Configuration & Constants ---
INR_TO_USD = 85.0  # Use float for division

# --- Helper Functions ---

def format_currency(amount, currency='INR'):
    """Formats large numbers into millions (mn) or billions (bn)."""
    if currency == 'USD':
        amount /= INR_TO_USD

    prefix = '₹' if currency == 'INR' else '$'

    if abs(amount) >= 1e9:
        return f"{prefix}{amount / 1e9:.1f} bn"
    elif abs(amount) >= 1e6:
        return f"{prefix}{amount / 1e6:.1f} mn"
    elif abs(amount) >= 1e3:
         # Add k for thousands if needed, or just format
         return f"{prefix}{amount:,.0f}" # Keep precision for smaller numbers
    else:
        return f"{prefix}{amount:,.2f}" # Show decimals for very small amounts

def format_units(units):
    """Formats large unit numbers."""
    if abs(units) >= 1e9:
        return f"{units / 1e9:.1f} bn"
    elif abs(units) >= 1e6:
        return f"{units / 1e6:.1f} mn"
    elif abs(units) >= 1e3:
        return f"{units:,.0f}"
    else:
        return f"{units}"

def parse_list_input(input_str, data_type=int):
    """Safely parses comma-separated or JSON-like list strings."""
    input_str = input_str.strip()
    if not input_str:
        return []
    try:
        # Try parsing as JSON first (handles [1, 2, 3])
        if input_str.startswith('[') and input_str.endswith(']'):
            data = json.loads(input_str)
            return [data_type(item) for item in data]
        # Otherwise, try splitting by comma (handles 1, 2, 3)
        else:
            data = [item.strip() for item in input_str.split(',')]
            return [data_type(item) for item in data]
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        st.error(f"Error parsing list input '{input_str}': {e}. Please use format like [1, 2, 3] or 1, 2, 3.")
        return None # Indicate error

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("📈 Interactive Financial Projection Tool (Month-by-Month)")

# --- Session State Initialization ---
# We use session state extensively to keep track of progress and data
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'current_month' not in st.session_state:
    st.session_state.current_month = 1
if 'current_capital' not in st.session_state:
    st.session_state.current_capital = 0.0
if 'current_fixed_costs' not in st.session_state:
    st.session_state.current_fixed_costs = 0.0
if 'current_profit_per_unit' not in st.session_state: # Based on SellingPrice - VarCost
    st.session_state.current_profit_per_unit = 0.0
if 'config' not in st.session_state:
    st.session_state.config = {} # To store initial parameters

# --- Sidebar for Initial Configuration ---
st.sidebar.header("Initial Business Setup")

# Disable inputs if simulation has started
config_disabled = st.session_state.simulation_started

initial_capital = st.sidebar.number_input("Initial Capital (₹)", value=200000000, min_value=0.0, step=100000.0, disabled=config_disabled)
fixed_costs_initial = st.sidebar.number_input("Initial Monthly Fixed Costs (₹)", value=100000, min_value=0.0, step=10000.0, disabled=config_disabled)
variable_cost_per_unit = st.sidebar.number_input("Variable Cost Per Unit (₹)", value=13.0, min_value=0.0, step=0.5, format="%.2f", disabled=config_disabled)
selling_price_per_unit = st.sidebar.number_input("Selling Price Per Unit (₹)", value=18.0, min_value=0.0, step=0.5, format="%.2f", disabled=config_disabled)

st.sidebar.markdown("---")
st.sidebar.subheader("Growth & Limits")
profit_growth_rate_pct = st.sidebar.number_input("Profit Per Unit Growth Rate (% per month)", value=-1.0, step=0.5, format="%.2f", disabled=config_disabled)
fixed_cost_growth_rate_pct = st.sidebar.number_input("Fixed Cost Growth Rate (% per month)", value=100.0, step=1.0, format="%.1f", disabled=config_disabled)
fixed_cost_cap = st.sidebar.number_input("Fixed Cost Cap (₹)", value=999999999.0, min_value=0.0, step=100000.0, disabled=config_disabled)
diseconomies_scale_pct = st.sidebar.number_input("Diseconomies of Scale (% of Potential Revenue)", value=0.5, min_value=0.0, max_value=100.0, step=0.1, format="%.2f", disabled=config_disabled)
total_months = st.sidebar.number_input("Total Simulation Months", value=36, min_value=1, step=1, disabled=config_disabled)

st.sidebar.markdown("---")
st.sidebar.subheader("Capital Injections")
ci_months_str = st.sidebar.text_input("Capital Injection Months (e.g., [9,18,27] or 9, 18, 27)", "[9,18,27]", disabled=config_disabled)
ci_amounts_str = st.sidebar.text_input("Capital Injection Amounts (e.g., [2e8, 2e8, -99999])", "[200000000, 200000000, -99999]", disabled=config_disabled)

# --- Start Simulation Button ---
if not st.session_state.simulation_started:
    if st.sidebar.button("🚀 Start Simulation"):
        # Parse list inputs carefully
        ci_months = parse_list_input(ci_months_str, int)
        ci_amounts = parse_list_input(ci_amounts_str, float) # Use float for amounts

        if ci_months is None or ci_amounts is None:
            st.error("Invalid format for Capital Injection lists. Please correct and try again.")
        elif len(ci_months) != len(ci_amounts):
            st.error("Capital Injection Months and Amounts must have the same number of entries.")
        elif selling_price_per_unit <= variable_cost_per_unit:
             st.warning("Warning: Selling Price per Unit is not greater than Variable Cost per Unit. Initial profit margin is non-positive.")
             # Allow simulation to proceed but warn user.
             st.session_state.simulation_started = True # Still start
        else:
            st.session_state.simulation_started = True
            st.session_state.current_month = 1
            st.session_state.current_capital = float(initial_capital)
            st.session_state.current_fixed_costs = float(fixed_costs_initial)
            st.session_state.current_profit_per_unit = float(selling_price_per_unit - variable_cost_per_unit)
            st.session_state.results_df = pd.DataFrame() # Reset results

            # Store config
            st.session_state.config = {
                'variable_cost_per_unit': float(variable_cost_per_unit),
                'selling_price_per_unit': float(selling_price_per_unit),
                'profit_growth_rate': profit_growth_rate_pct / 100.0,
                'fixed_cost_growth_rate': fixed_cost_growth_rate_pct / 100.0,
                'fixed_cost_cap': float(fixed_cost_cap),
                'diseconomies_scale_rate': diseconomies_scale_pct / 100.0,
                'total_months': int(total_months),
                'ci_map': dict(zip(ci_months, ci_amounts)) # Easier lookup
            }
            st.rerun() # Rerun to move to the monthly input stage

# --- Main Simulation Area ---
if st.session_state.simulation_started:

    config = st.session_state.config
    current_month = st.session_state.current_month

    if current_month > config['total_months']:
        st.success(f"🎉 Simulation Complete for {config['total_months']} months!")
        st.balloons()
    else:
        st.header(f"Month {current_month} / {config['total_months']}")
        st.subheader("Monthly Operational Inputs")
        st.markdown(f"**Starting Capital:** {format_currency(st.session_state.current_capital, 'INR')} | **Current Fixed Costs:** {format_currency(st.session_state.current_fixed_costs, 'INR')}")

        with st.form(key=f"month_{current_month}_form"):
            # --- New Inputs Per Month ---
            units_manufactured = st.number_input("Units Manufactured this Month", min_value=0, step=100, key=f"units_m_{current_month}")
            shops_sold_to = st.number_input("Shops Sold To this Month", min_value=0, step=1, key=f"shops_{current_month}")
            money_received = st.number_input("Actual Cash Received this Month (₹)", min_value=0.0, step=1000.0, format="%.2f", key=f"cash_{current_month}")

            submit_button = st.form_submit_button(label=f"Process Month {current_month}")

            if submit_button:
                # --- Calculations for the Current Month ---
                starting_capital_month = st.session_state.current_capital
                fixed_costs_month = st.session_state.current_fixed_costs
                var_cost_per_unit = config['variable_cost_per_unit']

                # Calculate costs for this month
                variable_costs_total = units_manufactured * var_cost_per_unit
                total_operational_costs = fixed_costs_month + variable_costs_total

                # Potential Revenue (useful for diseconomies and reference)
                # Use the *current* selling price, which reflects profit growth from *previous* months
                current_selling_price = var_cost_per_unit + st.session_state.current_profit_per_unit
                potential_revenue = units_manufactured * current_selling_price

                # Diseconomies of Scale Cost
                diseconomies_cost = potential_revenue * config['diseconomies_scale_rate']

                # Total Costs including Diseconomies
                total_costs_final = total_operational_costs + diseconomies_cost

                # Profit for the month (Cash based)
                profit_month = money_received - total_costs_final # Cash Received minus ALL costs

                # Check for Capital Injection
                capital_injection = config['ci_map'].get(current_month, 0.0)

                # Ending Capital
                ending_capital = starting_capital_month + profit_month + capital_injection

                # Calculate Profit Margin % (based on potential per-unit profit)
                # Avoid division by zero if selling price somehow becomes zero
                profit_margin_percentage = 0.0
                if current_selling_price > 0:
                    profit_margin_percentage = (st.session_state.current_profit_per_unit / current_selling_price) * 100


                # --- Store Results ---
                month_data = {
                    'Month': current_month,
                    'Starting Capital (INR)': starting_capital_month,
                    'Fixed Costs (INR)': fixed_costs_month,
                    'Units Manufactured': units_manufactured,
                    'Variable Costs (INR)': variable_costs_total,
                    'Potential Revenue (INR)': potential_revenue, # Based on units * selling price
                    'Money Received (INR)': money_received,      # Actual cash inflow
                    'Diseconomies Cost (INR)': diseconomies_cost,
                    'Total Costs (INR)': total_costs_final, # Fixed + Var + Diseconomies
                    'Profit (INR)': profit_month,           # Cash based profit
                    'Profit Margin (%)': profit_margin_percentage, # Based on potential unit profit
                    'Capital Injection (INR)': capital_injection,
                    'Ending Capital (INR)': ending_capital,
                    'Shops Sold To': shops_sold_to,
                    # Store raw values for charting/conversion
                    '_start_cap': starting_capital_month,
                    '_fixed_costs': fixed_costs_month,
                    '_revenue': potential_revenue, # Using potential for consistency
                    '_profit': profit_month,
                    '_diseconomies': diseconomies_cost,
                    '_injection': capital_injection,
                    '_end_cap': ending_capital,
                }

                new_results_df = pd.DataFrame([month_data])
                st.session_state.results_df = pd.concat([st.session_state.results_df, new_results_df], ignore_index=True)


                # --- Update State for Next Month ---
                st.session_state.current_capital = ending_capital
                st.session_state.current_fixed_costs = min(fixed_costs_month * (1 + config['fixed_cost_growth_rate']), config['fixed_cost_cap'])
                st.session_state.current_profit_per_unit *= (1 + config['profit_growth_rate'])
                st.session_state.current_month += 1

                st.rerun() # Rerun to display updated table/chart and prompt for next month


    # --- Display Results Table and Chart ---
    if not st.session_state.results_df.empty:
        st.subheader("Simulation Results")

        # Currency Toggle
        currency = st.radio("Display Currency", ('INR', 'USD'), index=0, horizontal=True, key="currency_toggle")

        # Prepare display dataframe
        display_df = st.session_state.results_df.copy()

        # Select columns to display and format
        cols_to_display = [
            'Month', 'Starting Capital', 'Fixed Costs', 'Units Manufactured',
            'Potential Revenue', 'Money Received', 'Profit Margin (%)', 'Profit',
            'Diseconomies Cost', 'Capital Injection', 'Ending Capital', 'Shops Sold To'
        ]
         # Rename columns for display clarity and apply formatting based on currency
        display_df_formatted = pd.DataFrame()
        display_df_formatted['Month'] = display_df['Month']
        display_df_formatted['Starting Capital'] = display_df['_start_cap'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Fixed Costs'] = display_df['_fixed_costs'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Units Manufactured'] = display_df['Units Manufactured'].apply(format_units)
        display_df_formatted['Potential Revenue'] = display_df['_revenue'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Money Received'] = display_df['Money Received (INR)'].apply(lambda x: format_currency(x, currency)) # Base is INR
        display_df_formatted['Profit Margin (%)'] = display_df['Profit Margin (%)'].apply(lambda x: f"{x:.1f}%")
        display_df_formatted['Profit'] = display_df['_profit'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Diseconomies Cost'] = display_df['_diseconomies'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Capital Injection'] = display_df['_injection'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Ending Capital'] = display_df['_end_cap'].apply(lambda x: format_currency(x, currency))
        display_df_formatted['Shops Sold To'] = display_df['Shops Sold To'].apply(format_units)


        st.dataframe(display_df_formatted, hide_index=True, use_container_width=True)

        # --- Charting ---
        st.subheader("Financial Trends")

        # Prepare data for chart (use raw values, convert if USD)
        chart_df = display_df[['Month']].copy()
        chart_df['Starting Capital'] = display_df['_start_cap']
        chart_df['Fixed Costs'] = display_df['_fixed_costs']
        chart_df['Potential Revenue'] = display_df['_revenue'] # Chart potential revenue
        chart_df['Profit'] = display_df['_profit']
        chart_df['Diseconomies of Scale'] = display_df['_diseconomies']
        chart_df['Capital Injection'] = display_df['_injection']
        chart_df['Ending Capital'] = display_df['_end_cap']

        if currency == 'USD':
            for col in ['Starting Capital', 'Fixed Costs', 'Potential Revenue', 'Profit', 'Diseconomies of Scale', 'Capital Injection', 'Ending Capital']:
                chart_df[col] = chart_df[col] / INR_TO_USD

        # Melt dataframe for Plotly
        chart_df_melt = chart_df.melt(id_vars=['Month'], var_name='Metric', value_name='Amount')

        # Create Plotly chart
        fig = px.line(
            chart_df_melt,
            x='Month',
            y='Amount',
            color='Metric',
            title=f'Financial Trends Over Time ({currency})',
            labels={'Amount': f'Amount ({currency})', 'Month': 'Month'},
            markers=True # Add markers to see individual points
        )

        fig.update_layout(legend_title_text='Metrics')
        st.plotly_chart(fig, use_container_width=True)

elif not st.session_state.simulation_started:
    st.info("ℹ️ Configure the initial parameters in the sidebar and click 'Start Simulation'.")

# --- Add Reset Button ---
if st.session_state.simulation_started:
    if st.sidebar.button("🔄 Reset Simulation"):
        # Clear relevant session state keys
        keys_to_reset = ['simulation_started', 'results_df', 'current_month',
                         'current_capital', 'current_fixed_costs',
                         'current_profit_per_unit', 'config']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        # Optionally clear widget states if needed, though rerun usually handles it
        # st.experimental_rerun() # Use st.rerun() in newer versions
        st.rerun()
