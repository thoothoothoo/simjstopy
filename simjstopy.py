import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import ast
from copy import deepcopy

# --- Constants ---
INR_TO_USD = 85.0
# Define columns that represent currency values for formatting
CURRENCY_COLUMNS = [
    "Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
    "Variable Costs", "Gross Profit", "Diseconomies Cost", "Operating Profit",
    "Capital Injection", "Ending Capital", "Inventory Value", "Cash",
    "Total Assets", "Total Equity", "Cash from Operations", "Cash from Financing",
    "Net Change in Cash", "Selling Price Per Unit" # Selling price is also a currency value
]
# Define columns that are editable in the main table
EDITABLE_COLUMNS = ["Fixed Costs", "Capital Injection", "Selling Price Per Unit"]

# --- Helper Functions ---

def format_currency_display(amount, currency):
    """Formats currency for display only (with mn/bn suffixes)."""
    display_amount = amount
    symbol = "₹" if currency == "INR" else "$"
    if currency == 'USD':
        if INR_TO_USD == 0: return f"{symbol}NaN"
        # Prevent division by zero errors with large numbers if INR_TO_USD is invalid
        try:
            display_amount /= INR_TO_USD
        except (TypeError, ZeroDivisionError):
             return f"{symbol}NaN"


    # Handle potential non-numeric input gracefully
    if not isinstance(display_amount, (int, float)):
        return f"{symbol}Invalid"

    abs_amount = abs(display_amount)
    sign = "-" if display_amount < 0 else ""

    if abs_amount >= 1e9:
        val_str = f"{sign}{(abs_amount / 1e9):.1f} bn"
    elif abs_amount >= 1e6:
        val_str = f"{sign}{(abs_amount / 1e6):.1f} mn"
    elif abs_amount >= 1e3: # Add k for thousands
        val_str = f"{sign}{(abs_amount / 1e3):.1f} k"
    else:
        val_str = f"{sign}{display_amount:,.2f}" # Keep decimals for smaller numbers
    return f"{symbol}{val_str}"

# Removed format_currency_editor as complex formats didn't work. Use std formats.

def format_units(units):
    """Formats unit counts with mn/bn/k suffixes."""
    if not isinstance(units, (int, float)):
        return "Invalid"

    abs_units = abs(units)
    sign = "-" if units < 0 else ""

    if abs_units >= 1e9:
        return f"{sign}{(abs_units / 1e9):.1f} bn"
    elif abs_units >= 1e6:
        return f"{sign}{(abs_units / 1e6):.1f} mn"
    elif abs_units >= 1e3: # Add k for thousands
        return f"{sign}{(abs_units / 1e3):.1f} k"
    else:
        return f"{sign}{int(units):,}" if units == int(units) else f"{sign}{units:,.0f}" # No decimals for units

def parse_list_input(input_string, default_value):
    """Safely parses a string input expected to be a list."""
    # (Same as before)
    try:
        parsed_list = ast.literal_eval(input_string)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            st.warning(f"Input '{input_string}' is not a valid list. Using default: {default_value}")
            return default_value
    except (ValueError, SyntaxError, TypeError) as e:
        st.warning(f"Could not parse input '{input_string}' as a list: {e}. Using default: {default_value}")
        return default_value

# --- Core Calculation Function ---
# (calculate_projections remains largely the same logic as previous version)
def calculate_projections(
    initial_capital, fixed_costs_initial, variable_cost_per_unit,
    selling_price_per_unit_initial, sales_percentage, ci_months, ci_amounts,
    fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
    months, user_edits={} # Pass user edits as a dictionary {(month, col): value}
):
    """Calculates the financial projections incorporating inventory and user overrides."""

    # Convert percentages to decimals
    sales_rate = sales_percentage / 100.0
    fixed_cost_growth_rate = fixed_cost_growth_rate_input / 100.0
    diseconomies_of_scale = diseconomies_of_scale_input / 100.0

    capital_injection_map = dict(zip(ci_months, ci_amounts))

    data = []
    starting_capital = initial_capital
    current_fixed_costs = fixed_costs_initial
    inventory_units = 0 # Starting inventory

    # Validation
    if variable_cost_per_unit < 0:
        st.warning("Warning: Variable Cost Per Unit is negative.")

    for month in range(1, months + 1):

        # --- Apply User Edits/Overrides ---
        edited_fixed_costs = user_edits.get((month, "Fixed Costs"))
        effective_fixed_costs = edited_fixed_costs if edited_fixed_costs is not None else current_fixed_costs

        edited_capital_injection = user_edits.get((month, "Capital Injection"))
        initial_capital_injection = capital_injection_map.get(month, 0)
        effective_capital_injection = edited_capital_injection if edited_capital_injection is not None else initial_capital_injection

        edited_selling_price = user_edits.get((month, "Selling Price Per Unit"))
        effective_selling_price = edited_selling_price if edited_selling_price is not None else selling_price_per_unit_initial

        if effective_selling_price <= 0:
             #st.warning(f"Month {month}: Effective Selling Price is zero or negative. Revenue will be zero.")
             effective_selling_price = 0 # Prevent downstream errors
        # --- End Apply User Edits ---

        # 1. Funds available (after fixed costs)
        remaining_for_production = starting_capital - effective_fixed_costs

        # 2. Units Produced
        units_produced = 0
        if remaining_for_production > 0 and variable_cost_per_unit > 0:
             try: # Avoid overflow potential with huge numbers
                 units_produced = math.floor(remaining_for_production / variable_cost_per_unit)
             except OverflowError:
                 units_produced = 0 # Or some indicator of max capacity reached
                 st.warning(f"Month {month}: Potential overflow calculating units produced. Setting to 0.")
        elif remaining_for_production > 0 and variable_cost_per_unit <= 0:
             units_produced = 0 # Avoid infinite/undefined production

        # 3. Units Available & Units Sold
        units_available_for_sale = inventory_units + units_produced
        units_sold = math.floor(units_available_for_sale * sales_rate) # Apply sales rate
        ending_inventory = units_available_for_sale - units_sold

        # 4. Costs
        variable_costs_total = units_produced * variable_cost_per_unit # Cost based on production

        # 5. Revenue
        revenue = units_sold * effective_selling_price

        # 6. Diseconomies Cost
        diseconomies_cost = revenue * diseconomies_of_scale

        # 7. Profit Calculation (Standard Accounting)
        # Gross Profit based on COGS = units_sold * variable_cost_per_unit
        cogs = units_sold * variable_cost_per_unit
        gross_profit = revenue - cogs
        operating_profit = gross_profit - effective_fixed_costs - diseconomies_cost


        # 8. Profit Margin (based on effective price)
        profit_margin_percentage = 0.0
        if effective_selling_price > 0: # Avoid division by zero
            margin_per_unit = effective_selling_price - variable_cost_per_unit
            profit_margin_percentage = (margin_per_unit / effective_selling_price) * 100

        # 9. Ending Capital (Cash Balance)
        # Cash flow: StartCash + Revenue - FixedCosts - VariableCosts_Produced - Diseconomies + Injection
        # **Correction**: VC cash outflow happens when units are *produced*
        ending_capital = (
            starting_capital
            + revenue
            - effective_fixed_costs
            - variable_costs_total # Cash out for production costs this month
            - diseconomies_cost
            + effective_capital_injection
        )

        # 10. Inventory Valuation (using Variable Cost)
        inventory_value = ending_inventory * variable_cost_per_unit


        month_data = {
            "Month": month,
            "Starting Capital": starting_capital, # Cash at start
            "Fixed Costs": effective_fixed_costs,
            "Remaining for Prod.": max(0, remaining_for_production),
            "Units Produced": units_produced,
            "Variable Costs": variable_costs_total, # Cost of production incurred
            "Inventory Units Start": inventory_units,
            "Units Available": units_available_for_sale,
            "Units Sold": units_sold,
            "Inventory Units End": ending_inventory,
            "Inventory Value": inventory_value, # Asset value
            "Selling Price Per Unit": effective_selling_price, # Price used
            "Revenue": revenue,
            "Gross Profit": gross_profit, # Rev - COGS
            "Diseconomies Cost": diseconomies_cost,
            "Operating Profit": operating_profit, # GP - FC - Diseconomies
            "Profit Margin (%)": profit_margin_percentage, # Based on unit economics
            "Capital Injection": effective_capital_injection,
            "Ending Capital": ending_capital, # Cash at end
        }
        data.append(month_data)

        # --- Update for Next Month ---
        starting_capital = ending_capital # Ending cash becomes starting cash
        inventory_units = ending_inventory # Ending inventory becomes starting inventory

        # Update Fixed Costs (only if not overridden this month)
        # Apply growth rate based on the fixed cost actually used this month
        base_fc_for_next_month = effective_fixed_costs
        if base_fc_for_next_month < fixed_cost_cap :
            current_fixed_costs = min(base_fc_for_next_month * (1 + fixed_cost_growth_rate), fixed_cost_cap)
        else:
            current_fixed_costs = fixed_cost_cap


    return pd.DataFrame(data)


# --- Aggregation & Summary Functions ---

def generate_summary_tables(df_monthly, period='Quarterly', vc_per_unit=0):
    """Generates P&L, Balance Sheet, Cash Flow summaries."""
    # *** FIX: Handle empty input DataFrame ***
    if df_monthly.empty:
        st.warning("No monthly data available to generate summaries.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try: # Wrap in try-except for robustness during aggregation
        df = df_monthly.copy()
        df['Year'] = (df['Month'] - 1) // 12 + 1
        df['Quarter'] = (df['Month'] - 1) // 3 + 1

        group_col = 'Year' if period == 'Annual' else ['Year', 'Quarter']
        agg_funcs = {
            # P&L Items (Flows - Sum)
            'Revenue': 'sum',
            'Gross Profit': 'sum', # Sum monthly GP
            'Fixed Costs': 'sum', # Sum monthly FC
            'Diseconomies Cost': 'sum', # Sum monthly DC
            'Operating Profit': 'sum', # Sum monthly OP
            'Capital Injection': 'sum', # Sum injections
            'Variable Costs': 'sum', # Sum monthly production VC
            # Balance Sheet Items (Stocks - Last Value)
            'Ending Capital': 'last', # Cash balance at end of period
            'Inventory Value': 'last', # Inventory value at end of period
            # Cash Flow Items
            'Starting Capital': 'first', # Cash at start of period
        }

        # Perform aggregation
        summary = df.groupby(group_col).agg(agg_funcs).reset_index()

        # --- P&L Summary ---
        pnl_cols = ['Revenue', 'Gross Profit', 'Fixed Costs', 'Diseconomies Cost', 'Operating Profit']
        pnl = summary[['Year', 'Quarter' if period == 'Quarterly' else 'Year'] + pnl_cols].copy()
        pnl.rename(columns={'Fixed Costs': 'Total Fixed Costs'}, inplace=True)

        # --- Balance Sheet Summary (Simplified) ---
        bs_cols = ['Ending Capital', 'Inventory Value']
        bs = summary[['Year', 'Quarter' if period == 'Quarterly' else 'Year'] + bs_cols].copy()
        bs.rename(columns={'Ending Capital': 'Cash'}, inplace=True)
        bs['Total Assets'] = bs['Cash'] + bs['Inventory Value']
        # Assuming Equity = Assets (no liabilities modeled)
        bs['Total Equity'] = bs['Total Assets']
        bs_cols = ['Cash', 'Inventory Value', 'Total Assets', 'Total Equity'] # Reorder for display
        bs = bs[['Year', 'Quarter' if period == 'Quarterly' else 'Year'] + bs_cols]

        # --- Cash Flow Summary (Simplified) ---
        cf_cols = ['Starting Capital', 'Ending Capital', 'Operating Profit', 'Capital Injection', 'Variable Costs']
        cf = summary[['Year', 'Quarter' if period == 'Quarterly' else 'Year'] + cf_cols].copy()

        # Cash from Ops = Operating Profit (very simplified - adjust for non-cash & WC changes if needed)
        # We don't have depreciation. Change in Inventory is a key adjustment needed.
        # Change in Inventory Value = Ending Inventory Value (from BS) - Starting Inventory Value (need previous period's end)
        bs['Prev Inventory Value'] = bs['Inventory Value'].shift(1).fillna(0) # Get previous period's ending inventory
        cf = pd.merge(cf, bs[['Year', 'Quarter' if period == 'Quarterly' else 'Year', 'Inventory Value', 'Prev Inventory Value']], on=['Year', 'Quarter' if period == 'Quarterly' else 'Year'], how='left')
        cf['Change in Inventory'] = cf['Inventory Value'] - cf['Prev Inventory Value']
        # Simplified CFO = OP - Change in Inventory (increase in inventory reduces cash)
        cf['Cash from Operations'] = cf['Operating Profit'] - cf['Change in Inventory']

        cf['Cash from Investing'] = 0 # No investing activities modeled
        cf['Cash from Financing'] = cf['Capital Injection']
        cf['Net Change in Cash'] = cf['Cash from Operations'] + cf['Cash from Investing'] + cf['Cash from Financing']
        # Verification: Net Change should equal Ending Capital - Starting Capital
        # cf['Verify Net Change'] = cf['Ending Capital'] - cf['Starting Capital']

        cf_cols_final = ['Cash from Operations', 'Cash from Investing', 'Cash from Financing', 'Net Change in Cash', 'Ending Capital']
        cf = cf[['Year', 'Quarter' if period == 'Quarterly' else 'Year'] + cf_cols_final]

        # --- Add Period Column for Charting ---
        # *** FIX: Check if DataFrame is empty before adding Period ***
        if not pnl.empty:
            if period == 'Quarterly':
                pnl['Period'] = pnl['Year'].astype(str) + '-Q' + pnl['Quarter'].astype(str)
                bs['Period'] = bs['Year'].astype(str) + '-Q' + bs['Quarter'].astype(str)
                cf['Period'] = cf['Year'].astype(str) + '-Q' + cf['Quarter'].astype(str)
            else:
                 # Ensure Year column exists before trying to access it
                if 'Year' in pnl.columns:
                     pnl['Period'] = pnl['Year'].astype(str)
                if 'Year' in bs.columns:
                     bs['Period'] = bs['Year'].astype(str)
                if 'Year' in cf.columns:
                     cf['Period'] = cf['Year'].astype(str)
        else:
             # If empty, create empty Period column to avoid errors later
             pnl['Period'] = []
             bs['Period'] = []
             cf['Period'] = []


        return pnl, bs, cf

    except Exception as e:
        st.error(f"Error generating summary tables: {e}")
        # Return empty dataframes on error
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("Financial Projection Tool (Advanced V2)")

# --- Initialize Session State ---
if 'df_results' not in st.session_state: st.session_state.df_results = pd.DataFrame()
if 'user_edits' not in st.session_state: st.session_state.user_edits = {}
if 'editor_key' not in st.session_state: st.session_state.editor_key = 0
if 'last_params' not in st.session_state: st.session_state.last_params = {}

# --- Input Form (Sidebar) ---
with st.sidebar:
    st.header("Initial Parameters")
    initial_capital = st.number_input("Initial Capital:", min_value=0.0, value=200000000.0, step=1000000.0, format="%.2f")
    fixed_costs_initial = st.number_input("Initial Fixed Costs:", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    variable_cost_per_unit = st.number_input("Variable Cost Per Unit:", min_value=0.0, value=13.0, step=0.5, format="%.2f")
    selling_price_per_unit_initial = st.number_input("Initial Selling Price Per Unit:", min_value=0.01, value=18.0, step=0.5, format="%.2f")
    sales_percentage = st.number_input("Sales Percentage (% available/mo):", min_value=0.0, max_value=100.0, value=90.0, step=1.0, format="%.1f")

    st.header("Growth & Constraints")
    fixed_cost_growth_rate_input = st.number_input("Fixed Cost Growth Rate (%/mo):", value=100.0, step=1.0, format="%.2f")
    fixed_cost_cap = st.number_input("Fixed Cost Cap:", min_value=0.0, value=999999999.0, step=10000.0, format="%.2f")
    diseconomies_of_scale_input = st.number_input("Diseconomies (% of Revenue):", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    months = st.number_input("Number of Months:", min_value=1, value=36, step=1)

    st.header("Capital Injections")
    ci_months_str = st.text_input("Injection Months (e.g., [9,18,27]):", value="[9,18,27]")
    ci_amounts_str = st.text_input("Injection Amounts (e.g., [200M,200M,-99k]):", value="[200000000,200000000,-99999]")

    # --- Actions ---
    col1, col2 = st.columns(2)
    generate_button = col1.button("Generate Projections", type="primary", use_container_width=True)
    clear_edits_button = col2.button("Clear Edits & Recalculate", use_container_width=True)

    # --- Display Currency Toggle ---
    st.header("Display")
    current_currency = st.radio("Currency:", ("INR", "USD"), index=0, horizontal=True, key="currency_toggle") # Add key


# --- Parse Inputs & Handle Actions ---
# (Input parsing and validation logic remains similar)
ci_months = parse_list_input(ci_months_str, [9, 18, 27])
ci_amounts = parse_list_input(ci_amounts_str, [200000000, 200000000, -99999])

valid_inputs = True
if len(ci_months) != len(ci_amounts):
    st.sidebar.error("Error: Injection months/amounts lists length mismatch.")
    valid_inputs = False
if variable_cost_per_unit < 0:
     st.sidebar.warning("Warning: Variable cost is negative.")
if selling_price_per_unit_initial <= 0:
     st.sidebar.error("Error: Initial Selling Price must be positive.")
     valid_inputs = False

current_params = { # Store params used for calculation
    "ic": initial_capital, "fci": fixed_costs_initial, "vcpu": variable_cost_per_unit,
    "spui": selling_price_per_unit_initial, "sp": sales_percentage, "cim": ci_months, "cia": ci_amounts,
    "fcgr": fixed_cost_growth_rate_input, "fcc": fixed_cost_cap, "dos": diseconomies_of_scale_input,
    "m": months
}

# --- Calculation Trigger Logic ---
# (Remains similar, checks buttons, param changes)
needs_recalculation = False
trigger_source = None # For debugging/info

if generate_button:
    st.session_state.user_edits = {}
    needs_recalculation = True
    st.session_state.editor_key += 1
    trigger_source = "Generate Button"
elif clear_edits_button:
    if st.session_state.user_edits:
        st.session_state.user_edits = {}
        needs_recalculation = True
        st.sidebar.success("Edits cleared.")
        st.session_state.editor_key += 1
        trigger_source = "Clear Edits Button"
    else:
        st.sidebar.info("No user edits to clear.")
elif st.session_state.df_results is not None and not st.session_state.df_results.empty and current_params != st.session_state.get('last_params', {}):
     #st.sidebar.info("Parameters changed. Recalculating...") # Can be noisy
     st.session_state.user_edits = {} # Clear edits on param change
     needs_recalculation = True
     st.session_state.editor_key += 1
     trigger_source = "Parameter Change"

if needs_recalculation and valid_inputs:
    #st.info(f"Recalculating projections... Triggered by: {trigger_source}")
    st.session_state.df_results = calculate_projections(
        initial_capital, fixed_costs_initial, variable_cost_per_unit,
        selling_price_per_unit_initial, sales_percentage, ci_months, ci_amounts,
        fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
        months, st.session_state.user_edits
    )
    st.session_state.last_params = current_params


# --- Display Monthly Editable Table ---
if 'df_results' in st.session_state and not st.session_state.df_results.empty:

    st.header("Monthly Projections")
    currency_symbol = "₹" if current_currency == "INR" else "$"
    st.caption(f"Editable Columns: {', '.join(EDITABLE_COLUMNS)}. Currency: {current_currency}")

    df_editable = st.session_state.df_results.copy()

    # --- Configure Columns for Editor ---
    # *** FIX: Use standard formats, add currency to label ***
    column_config = {}
    for col in df_editable.columns:
        label = col.replace("_", " ") # Basic label formatting
        is_editable = col in EDITABLE_COLUMNS
        is_currency = col in CURRENCY_COLUMNS
        is_unit = "Units" in col # Simple check for unit columns

        fmt = None
        if is_currency:
            fmt = "%.2f" # Standard float format for currency
            if is_editable: label = f"{label} ({currency_symbol}) (Editable)"
            else: label = f"{label} ({currency_symbol})"
        elif is_unit:
            fmt = "%d" # Integer format for units
        elif col == "Profit Margin (%)":
             fmt = "%.2f%%"

        column_config[col] = st.column_config.NumberColumn(
            label=label,
            format=fmt, # Use standard format
            disabled=not is_editable,
            # Add validation if needed (e.g., min_value)
            min_value=0 if col in ["Fixed Costs", "Selling Price Per Unit"] else None
        )

    # Specific override for Month (not a NumberColumn maybe?)
    column_config["Month"] = st.column_config.Column(label="Month", disabled=True)


    editor_instance_key = f"data_editor_{st.session_state.editor_key}"
    edited_data = st.data_editor(
        df_editable,
        column_config=column_config,
        key=editor_instance_key,
        hide_index=True,
        num_rows="fixed",
        use_container_width=True,
        height=400 # Set a fixed height for the monthly table
    )

    # --- Detect and Process Edits ---
    # (Edit detection logic remains similar, using editor state)
    editor_state = st.session_state.get(editor_instance_key, {})
    edited_rows_dict = editor_state.get("edited_rows", {})
    edits_found_in_editor = False
    potential_new_edits = {}

    if edited_rows_dict:
        # Create lookup only if results exist
        original_lookup = None
        if not st.session_state.df_results.empty and 'Month' in st.session_state.df_results.columns:
             original_lookup = st.session_state.df_results.set_index('Month')

        for row_index, changed_cells in edited_rows_dict.items():
             if row_index < len(st.session_state.df_results): # Check bounds
                 month = st.session_state.df_results.loc[row_index, "Month"]
                 for col_name, new_value in changed_cells.items():
                     if col_name in EDITABLE_COLUMNS:
                         # Validation
                         is_valid_edit = True
                         # Handle potential non-numeric values from editor if format fails
                         if not isinstance(new_value, (int, float)):
                              st.warning(f"Invalid input type '{type(new_value)}' for {col_name} in month {month}. Ignoring edit.")
                              is_valid_edit = False
                         elif col_name == "Fixed Costs" and new_value < 0:
                              st.warning(f"Month {month}: Fixed Costs cannot be negative. Change ignored.")
                              is_valid_edit = False
                         elif col_name == "Selling Price Per Unit" and new_value <= 0:
                              st.warning(f"Month {month}: Selling Price must be positive. Change ignored.")
                              is_valid_edit = False

                         if is_valid_edit and original_lookup is not None:
                             try:
                                 # Use original df_results for comparison base
                                 original_value_from_calc = original_lookup.loc[month, col_name]
                                 current_override = st.session_state.user_edits.get((month, col_name))
                                 value_to_compare = current_override if current_override is not None else original_value_from_calc

                                 # Check if editor value is meaningfully different (allow for float precision)
                                 if value_to_compare is None or not math.isclose(new_value, value_to_compare, rel_tol=1e-6):
                                     potential_new_edits[(month, col_name)] = new_value
                                     edits_found_in_editor = True
                             except KeyError:
                                 st.warning(f"Original value lookup failed for Month {month}, {col_name}.")
                             except TypeError as e:
                                 st.error(f"Type error comparing edit for {col_name} M{month}: {e}. New: {new_value}, Old: {value_to_compare}")


    # If valid edits found, update state and rerun
    if edits_found_in_editor and valid_inputs:
         changed_edits = False
         for k, v in potential_new_edits.items():
             # Check against existing edits, considering float tolerance
             existing_edit = st.session_state.user_edits.get(k)
             if existing_edit is None or not math.isclose(v, existing_edit, rel_tol=1e-6):
                 st.session_state.user_edits[k] = v
                 changed_edits = True

         if changed_edits:
             #st.info("Edits detected, rerunning calculation...") # Can be noisy
             st.rerun()


    # --- Display Formatted Monthly Table (Read-Only) ---
    # *** FIX: Reinstate this for easier visual scanning with mn/bn/k ***
    st.header("Formatted Monthly View (Read-Only)")
    st.caption("This view uses k/mn/bn suffixes for large numbers.")
    df_display = st.session_state.df_results.copy()

    # Identify columns to format
    cols_to_format_currency = [col for col in df_display.columns if col in CURRENCY_COLUMNS]
    cols_to_format_units = [col for col in df_display.columns if "Units" in col] # Includes inventory, produced, sold etc.

    for col in cols_to_format_currency:
         df_display[col] = df_display[col].apply(lambda x: format_currency_display(x, current_currency))
    for col in cols_to_format_units:
         df_display[col] = df_display[col].apply(format_units)
    if "Profit Margin (%)" in df_display.columns:
        df_display["Profit Margin (%)"] = df_display["Profit Margin (%)"].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A")


    # Define display order (optional)
    display_order = [
        "Month", "Starting Capital", "Fixed Costs", "Units Produced", "Variable Costs",
        "Units Sold", "Revenue", "Gross Profit", "Operating Profit",
        "Capital Injection", "Ending Capital", "Inventory Units End", "Inventory Value"
    ]
    display_cols = [col for col in display_order if col in df_display.columns] + \
                   [col for col in df_display.columns if col not in display_order] # Add any remaining columns

    st.dataframe(df_display[display_cols], hide_index=True, use_container_width=True)


    # --- Display Monthly Chart ---
    # (Monthly chart logic remains the same)
    st.header("Monthly Trends Chart")
    chart_df_monthly = st.session_state.df_results.copy()
    if current_currency == 'USD':
        for col in CURRENCY_COLUMNS:
            if col in chart_df_monthly.columns and INR_TO_USD != 0:
                try: # Add try-except for robustness
                    chart_df_monthly[col] = chart_df_monthly[col] / INR_TO_USD
                except TypeError:
                    chart_df_monthly[col] = 0 # Handle potential non-numeric data after edits?

    fig_monthly = go.Figure()
    plot_cols_monthly = {"Ending Capital": "Ending Capital (Cash)", "Revenue": "Revenue", "Operating Profit": "Operating Profit", "Inventory Value": "Inventory Value", "Capital Injection": "Capital Injection"}
    for col, name in plot_cols_monthly.items():
        if col in chart_df_monthly.columns:
             line_style = {}
             if "Profit" in col: line_style = dict(dash='dot')
             if "Inventory" in col: line_style = dict(dash='dash')
             fig_monthly.add_trace(go.Scatter(x=chart_df_monthly["Month"], y=chart_df_monthly[col], mode='lines', name=name, line=line_style))

    fig_monthly.update_layout(
        title="Key Monthly Financial Trends", xaxis_title="Month", yaxis_title=f"Amount ({current_currency})",
        legend_title="Metrics", hovermode="x unified"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.divider() # Add a visual separator

    # --- Quarterly / Annual Summaries ---
    st.header("Aggregated Financial Summaries")
    summary_period = st.radio("Select Summary Period:", ("Quarterly", "Annual"), index=0, horizontal=True, key="summary_period_toggle") # Add key

    # *** Ensure vc_per_unit is passed correctly ***
    pnl_summary, bs_summary, cf_summary = generate_summary_tables(
        st.session_state.df_results,
        period=summary_period,
        vc_per_unit=variable_cost_per_unit # Pass current VC for inventory valuation
        )

    # Display only if summaries are not empty
    if not pnl_summary.empty:
        pnl_display = pnl_summary.copy()
        bs_display = bs_summary.copy()
        cf_display = cf_summary.copy()

        pnl_currency_cols = pnl_display.select_dtypes(include='number').columns.drop(['Year', 'Quarter'], errors='ignore')
        bs_currency_cols = bs_display.select_dtypes(include='number').columns.drop(['Year', 'Quarter'], errors='ignore')
        cf_currency_cols = cf_display.select_dtypes(include='number').columns.drop(['Year', 'Quarter'], errors='ignore')

        for df_disp, cols in zip([pnl_display, bs_display, cf_display], [pnl_currency_cols, bs_currency_cols, cf_currency_cols]):
             for col in cols:
                 if col in df_disp.columns:
                     df_disp[col] = df_disp[col].apply(lambda x: format_currency_display(x, current_currency))

        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.subheader(f"{summary_period} Income Statement (P&L)")
            st.dataframe(pnl_display.drop(columns=['Year', 'Quarter'], errors='ignore'), hide_index=True, use_container_width=True)
            st.subheader(f"{summary_period} Cash Flow Statement")
            st.dataframe(cf_display.drop(columns=['Year', 'Quarter'], errors='ignore'), hide_index=True, use_container_width=True)
        with col_summary2:
            st.subheader(f"{summary_period} Balance Sheet")
            st.dataframe(bs_display.drop(columns=['Year', 'Quarter'], errors='ignore'), hide_index=True, use_container_width=True)

        # --- Summary Charts ---
        st.subheader(f"{summary_period} Financial Charts")
        # (Chart logic remains similar, ensure 'Period' column exists)
        if 'Period' in pnl_summary.columns:
             fig_pnl = px.bar(pnl_summary, x='Period', y=['Revenue', 'Operating Profit'], title=f'{summary_period} Revenue & Operating Profit', labels={'value': f'Amount ({current_currency})', 'variable': 'Metric'}, barmode='group')
             fig_pnl.update_layout(hovermode="x unified")
             st.plotly_chart(fig_pnl, use_container_width=True)
        if 'Period' in bs_summary.columns:
             fig_bs = px.bar(bs_summary, x='Period', y=['Cash', 'Inventory Value', 'Total Assets'], title=f'{summary_period} Assets Breakdown', labels={'value': f'Amount ({current_currency})', 'variable': 'Asset Type'}, barmode='group')
             fig_bs.update_layout(hovermode="x unified")
             st.plotly_chart(fig_bs, use_container_width=True)
        if 'Period' in cf_summary.columns:
             fig_cf = px.bar(cf_summary, x='Period', y=['Cash from Operations', 'Cash from Financing', 'Net Change in Cash'], title=f'{summary_period} Cash Flow Components', labels={'value': f'Amount ({current_currency})', 'variable': 'Cash Flow Type'}, barmode='group')
             fig_cf.update_layout(hovermode="x unified")
             st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.info(f"Not enough data to display {summary_period} summaries.")

else:
    # Initial state or after error preventing df_results generation
    if valid_inputs:
         st.info("Click 'Generate Projections' in the sidebar to start.")
    else:
         st.warning("Please correct the errors in the sidebar inputs before generating.")

# --- Optional Debug Info ---
# (Debug expander can be uncommented if needed)
# with st.expander("Debug Info"):
#     st.write("User Edits:", st.session_state.get('user_edits', {}))
#     st.write("Last Params:", st.session_state.get('last_params', {}))
#     st.write("Editor Key:", st.session_state.get('editor_key', 0))
#     st.write("Current Params:", current_params)
