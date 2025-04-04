import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import ast
from copy import deepcopy

# --- Constants ---
INR_TO_USD = 85.0

# --- Helper Functions ---
# (format_currency, format_units, parse_list_input remain the same)
def format_currency(amount, currency):
    """Formats currency values with mn/bn suffixes based on selected currency."""
    display_amount = amount
    if currency == 'USD':
        # Avoid division by zero if INR_TO_USD is somehow 0
        if INR_TO_USD == 0: return f"{sign}NaN"
        display_amount /= INR_TO_USD

    abs_amount = abs(display_amount)
    sign = "-" if display_amount < 0 else ""

    if abs_amount >= 1e9:
        return f"{sign}{(abs_amount / 1e9):.1f} bn"
    elif abs_amount >= 1e6:
        return f"{sign}{(abs_amount / 1e6):.1f} mn"
    else:
        # Format with commas and 2 decimal places
        return f"{sign}{display_amount:,.2f}"

def format_units(units):
    """Formats unit counts with mn/bn suffixes."""
    abs_units = abs(units)
    sign = "-" if units < 0 else ""

    if abs_units >= 1e9:
        return f"{sign}{(abs_units / 1e9):.1f} bn"
    elif abs_units >= 1e6:
        return f"{sign}{(abs_units / 1e6):.1f} mn"
    else:
        return f"{sign}{int(units):,}" if units == int(units) else f"{sign}{units:,}"

def parse_list_input(input_string, default_value):
    """Safely parses a string input expected to be a list."""
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
# (calculate_projections remains the same as in the previous version)
def calculate_projections(
    initial_capital, fixed_costs_initial, variable_cost_per_unit,
    selling_price_per_unit, profit_growth_rate_input, ci_months, ci_amounts,
    fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
    months, user_edits={} # Pass user edits as a dictionary {(month, col): value}
):
    """Calculates the financial projections incorporating user overrides."""

    # Convert percentages to decimals
    profit_growth_rate = profit_growth_rate_input / 100.0
    fixed_cost_growth_rate = fixed_cost_growth_rate_input / 100.0
    diseconomies_of_scale = diseconomies_of_scale_input / 100.0

    # Create a dictionary for quick lookup of initial capital injections
    capital_injection_map = dict(zip(ci_months, ci_amounts))

    data = []
    starting_capital = initial_capital
    current_fixed_costs = fixed_costs_initial
    profit_per_unit_for_margin = selling_price_per_unit - variable_cost_per_unit if selling_price_per_unit > variable_cost_per_unit else 0 # Avoid negative theoretical profit

    # Validation moved outside loop for efficiency
    if selling_price_per_unit <= 0:
         st.error("Error: Selling Price Per Unit must be positive.")
         return pd.DataFrame() # Return empty DataFrame on critical error
    if variable_cost_per_unit < 0:
         st.warning("Warning: Variable Cost Per Unit is negative.")
         # Allow calculation to proceed but this is unusual


    for month in range(1, months + 1):

        # --- Apply User Edits/Overrides ---
        # Check if user edited Fixed Costs for this month
        edited_fixed_costs = user_edits.get((month, "Fixed Costs"))
        effective_fixed_costs = edited_fixed_costs if edited_fixed_costs is not None else current_fixed_costs

        # Check if user edited Capital Injection for this month
        initial_capital_injection = capital_injection_map.get(month, 0)
        edited_capital_injection = user_edits.get((month, "Capital Injection"))
        effective_capital_injection = edited_capital_injection if edited_capital_injection is not None else initial_capital_injection
        # --- End Apply User Edits ---


        # 1. Calculate funds available after fixed costs (using effective costs)
        remaining_for_production = starting_capital - effective_fixed_costs

        # 2. Calculate units produced
        units_produced = 0
        # Prevent division by zero or issues with non-positive costs/margins
        if remaining_for_production > 0 and variable_cost_per_unit > 0:
            units_produced = math.floor(remaining_for_production / variable_cost_per_unit)
        elif remaining_for_production > 0 and variable_cost_per_unit <= 0:
             # If VC is zero or negative, production isn't limited by it.
             # This scenario needs clearer business logic. Assume 0 for safety.
             units_produced = 0
             # st.warning(f"Month {month}: Non-positive Variable Cost, Units Produced set to 0.")


        # 3. Calculate Revenue
        revenue = units_produced * selling_price_per_unit

        # 4. Calculate Diseconomies Cost
        diseconomies_cost = revenue * diseconomies_of_scale

        # 5. Calculate Profit (JS Logic)
        profit_js_logic = revenue - (remaining_for_production + effective_fixed_costs + diseconomies_cost)

        # Calculate Standard Profit (for potential display/analysis)
        variable_costs_total = units_produced * variable_cost_per_unit
        profit_standard = revenue - effective_fixed_costs - variable_costs_total - diseconomies_cost
        profit_for_month = profit_js_logic # Use JS logic for ending capital

        # 6. Calculate Profit Margin Percentage
        profit_margin_percentage = 0.0
        # Ensure selling_price_per_unit is not zero before division
        if selling_price_per_unit != 0:
            profit_margin_percentage = (profit_per_unit_for_margin / selling_price_per_unit) * 100

        # 7. Capital Injection is now effective_capital_injection

        # 8. Calculate Ending Capital
        ending_capital = starting_capital + profit_for_month + effective_capital_injection

        # Store results for this month
        month_data = {
            "Month": month,
            "Starting Capital": starting_capital,
            "Fixed Costs": effective_fixed_costs, # Store the potentially edited value
            "Remaining for Prod.": max(0, remaining_for_production), # Show 0 if negative
            "Units Produced": units_produced,
            "Revenue": revenue,
            "Profit Margin (%)": profit_margin_percentage,
            "Profit (JS Logic)": profit_for_month,
            "Profit (Standard)": profit_standard,
            "Diseconomies Cost": diseconomies_cost,
            "Capital Injection": effective_capital_injection, # Store the potentially edited value
            "Ending Capital": ending_capital,
        }
        data.append(month_data)

        # --- Update for Next Month ---
        starting_capital = ending_capital
        # Base next month's fixed costs on the *previous calculated or edited* value
        prev_fixed_costs = effective_fixed_costs # Use the value actually used this month
        # Apply growth rate only if not at the cap
        if prev_fixed_costs < fixed_cost_cap :
             current_fixed_costs = min(prev_fixed_costs * (1 + fixed_cost_growth_rate), fixed_cost_cap)
        else:
             current_fixed_costs = fixed_cost_cap # Stay at cap if already reached

        # Update the theoretical profit per unit used for margin display
        # Ensure growth rate doesn't lead to nonsensical values if profit margin is already negative
        if (1 + profit_growth_rate) > 0 or profit_per_unit_for_margin > 0: # Allow reduction but not instant flip to large negative?
             profit_per_unit_for_margin *= (1 + profit_growth_rate)
        # Alternatively, cap the reduction: profit_per_unit_for_margin = max(0, profit_per_unit_for_margin * (1 + profit_growth_rate))


    return pd.DataFrame(data)


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("Financial Projection Tool (Editable V2)")

# --- Initialize Session State ---
if 'df_results' not in st.session_state:
    st.session_state.df_results = pd.DataFrame() # Holds the main calculation result
if 'user_edits' not in st.session_state:
    st.session_state.user_edits = {} # Holds {(month, col): value} overrides from user
if 'editor_key' not in st.session_state:
     st.session_state.editor_key = 0 # Key to force editor refresh if needed


# --- Input Form ---
with st.sidebar: # Move inputs to sidebar for more space
    st.header("Initial Parameters")
    # (Input fields remain the same - initial_capital, fixed_costs_initial, etc.)
    initial_capital = st.number_input("Initial Capital (₹):", min_value=0.0, value=200000000.0, step=1000000.0, format="%.2f")
    fixed_costs_initial = st.number_input("Initial Fixed Costs (₹):", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    variable_cost_per_unit = st.number_input("Variable Cost Per Unit (₹):", min_value=None, value=13.0, step=0.5, format="%.2f") # Allow negative VC? Reverted to None min_value
    selling_price_per_unit = st.number_input("Selling Price Per Unit (₹):", min_value=0.01, value=18.0, step=0.5, format="%.2f") # Set min_value slightly above 0
    profit_growth_rate_input = st.number_input("Profit Growth Rate (% per month):", value=-1.0, step=0.1, format="%.2f", help="Affects the theoretical 'Profit Margin (%)' display calculation month-over-month.")
    fixed_cost_growth_rate_input = st.number_input("Fixed Cost Growth Rate (% per month):", value=100.0, step=1.0, format="%.2f")
    fixed_cost_cap = st.number_input("Fixed Cost Cap (₹):", min_value=0.0, value=999999999.0, step=10000.0, format="%.2f")
    diseconomies_of_scale_input = st.number_input("Diseconomies of Scale (% of Revenue):", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    months = st.number_input("Number of Months:", min_value=1, value=36, step=1)

    st.header("Capital Injections")
    ci_months_str = st.text_input("Injection Months (e.g., [9,18,27]):", value="[9,18,27]")
    ci_amounts_str = st.text_input("Injection Amounts (₹) (e.g., [200M,200M,-99k]):", value="[200000000,200000000,-99999]")

    # --- Actions ---
    col1, col2 = st.columns(2)
    generate_button = col1.button("Generate Projections", type="primary", use_container_width=True)
    clear_edits_button = col2.button("Clear Edits & Recalculate", use_container_width=True)

    # --- Currency Toggle ---
    st.header("Display")
    current_currency = st.radio("Currency:", ("INR", "USD"), index=0, horizontal=True)


# --- Parse Inputs & Handle Actions ---
# Parse list inputs safely
ci_months = parse_list_input(ci_months_str, [9, 18, 27])
ci_amounts = parse_list_input(ci_amounts_str, [200000000, 200000000, -99999])

# Validate list lengths and other critical inputs
valid_inputs = True
if len(ci_months) != len(ci_amounts):
    st.sidebar.error("Error: Injection months and amounts lists must have the same length.")
    valid_inputs = False

if selling_price_per_unit <= 0:
     st.sidebar.error("Error: Selling Price Per Unit must be positive.")
     valid_inputs = False


# Store initial params used for the current df_results to detect changes
current_params = {
    "ic": initial_capital, "fci": fixed_costs_initial, "vcpu": variable_cost_per_unit,
    "spu": selling_price_per_unit, "pgr": profit_growth_rate_input, "cim": ci_months, "cia": ci_amounts,
    "fcgr": fixed_cost_growth_rate_input, "fcc": fixed_cost_cap, "dos": diseconomies_of_scale_input,
    "m": months
}
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}

# --- Calculation Logic ---
# Recalculate if:
# 1. Generate button pressed
# 2. Clear edits button pressed (and edits existed)
# 3. Initial parameters changed since last calculation
# 4. Edits were made in the data_editor (handled later)

needs_recalculation = False
if generate_button:
    st.session_state.user_edits = {} # Clear edits
    needs_recalculation = True
    st.session_state.editor_key += 1 # Force editor refresh
elif clear_edits_button:
    if st.session_state.user_edits:
        st.session_state.user_edits = {}
        needs_recalculation = True
        st.sidebar.success("Edits cleared.")
        st.session_state.editor_key += 1 # Force editor refresh
    else:
        st.sidebar.info("No user edits to clear.")
# Check if parameters changed IF a calculation has already run
elif st.session_state.df_results is not None and not st.session_state.df_results.empty and current_params != st.session_state.last_params:
     st.sidebar.warning("Parameters changed. Recalculating...")
     st.session_state.user_edits = {} # Clear edits when params change
     needs_recalculation = True
     st.session_state.editor_key += 1 # Force editor refresh

if needs_recalculation and valid_inputs:
    st.session_state.df_results = calculate_projections(
        initial_capital, fixed_costs_initial, variable_cost_per_unit,
        selling_price_per_unit, profit_growth_rate_input, ci_months, ci_amounts,
        fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
        months, st.session_state.user_edits # Pass current edits (empty if Generate/Clear/ParamChange)
    )
    st.session_state.last_params = current_params # Store params used for this calculation


# --- Display Editable Table ---
if 'df_results' in st.session_state and not st.session_state.df_results.empty:

    st.header("Editable Projections Table")
    st.caption("Edit 'Fixed Costs' or 'Capital Injection'. Changes recalculate subsequent months.")

    # Prepare dataframe for editing (use raw numbers from the last calculation)
    df_editable = st.session_state.df_results.copy()

    # Define column configurations
    column_config = {col: st.column_config.NumberColumn(disabled=True, format="%.2f") for col in df_editable.columns if col not in ["Month", "Units Produced"]}
    column_config["Month"] = st.column_config.NumberColumn(disabled=True)
    column_config["Units Produced"] = st.column_config.NumberColumn(disabled=True, format="%d") # Integer
    column_config["Profit Margin (%)"] = st.column_config.NumberColumn(disabled=True, format="%.2f%%") # Percentage
    # Editable columns
    column_config["Fixed Costs"] = st.column_config.NumberColumn(label="Fixed Costs (Editable)", min_value=0, format="%.2f", disabled=False)
    column_config["Capital Injection"] = st.column_config.NumberColumn(label="Capital Injection (Editable)", format="%.2f", disabled=False)


    # Use a key that changes when we want to force a reset (like after Generate/Clear)
    editor_instance_key = f"data_editor_{st.session_state.editor_key}"

    # Display the data editor widget
    edited_data = st.data_editor(
        df_editable,
        column_config=column_config,
        key=editor_instance_key, # Use the dynamic key
        hide_index=True,
        num_rows="fixed", # Keep rows fixed to avoid complexity with adding/deleting
        use_container_width=True
    )

    # --- Detect and Process Edits ---
    # Access the editor's state directly using its key
    editor_state = st.session_state.get(editor_instance_key, {})
    edited_rows_dict = editor_state.get("edited_rows", {})

    edits_found_in_editor = False
    potential_new_edits = {}

    if edited_rows_dict: # Check if the editor state reports any edits
        # Create a temporary lookup for original values for comparison
        # Use 'Month' as index for easy lookup
        if 'Month' in st.session_state.df_results.columns:
             original_lookup = st.session_state.df_results.set_index('Month')
        else:
             original_lookup = st.session_state.df_results.copy() # Fallback if 'Month' is index

        for row_index, changed_cells in edited_rows_dict.items():
            # Get the corresponding month from the original DataFrame using the row index
            if row_index < len(st.session_state.df_results):
                month = st.session_state.df_results.loc[row_index, "Month"]

                for col_name, new_value in changed_cells.items():
                    if col_name in ["Fixed Costs", "Capital Injection"]:
                        # Get original value for comparison
                        original_value = None
                        if month in original_lookup.index:
                             try:
                                original_value = original_lookup.loc[month, col_name]
                             except KeyError:
                                st.warning(f"Could not find original value for Month {month}, Col {col_name}") # Should not happen if index is correct
                                continue # Skip this edit if original can't be found

                        # Check if the edit is valid and actually different
                        is_valid_edit = True
                        if col_name == "Fixed Costs" and new_value < 0:
                             st.warning(f"Invalid Edit: Fixed Costs for Month {month} cannot be negative ({new_value}). Change not applied.")
                             is_valid_edit = False
                             # We might need to revert the UI state here, which is hard with st.data_editor
                             # Best approach is often to ignore the invalid edit and let the next rerun overwrite it.

                        # Compare with existing user_edits OR the original calculated value
                        current_override = st.session_state.user_edits.get((month, col_name))
                        value_to_compare = current_override if current_override is not None else original_value

                        # Use math.isclose for float comparison
                        if is_valid_edit and (value_to_compare is None or not math.isclose(new_value, value_to_compare)):
                            potential_new_edits[(month, col_name)] = new_value
                            edits_found_in_editor = True
            else:
                 st.warning(f"Edited row index {row_index} out of bounds.")


    # If valid edits were found via editor state, update user_edits and trigger rerun
    if edits_found_in_editor and valid_inputs:
        # Check if the potential edits actually change the current user_edits state
        if potential_new_edits != {k:v for k,v in st.session_state.user_edits.items() if k in potential_new_edits}: # Compare only relevant keys
             st.session_state.user_edits.update(potential_new_edits)
             # print("User edits updated:", st.session_state.user_edits) # Debug print
             st.rerun() # Trigger a full script rerun to recalculate and refresh

    # --- Display Formatted Table (Read-Only) ---
    if not st.session_state.df_results.empty:
        st.header("Formatted Projections (Read-Only)")
        df_display = st.session_state.df_results.copy() # Use the latest calculated data
        currency_symbol = "₹" if current_currency == "INR" else "$"
        amount_cols_display = ["Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
                               "Profit (JS Logic)", "Profit (Standard)", "Diseconomies Cost", "Capital Injection", "Ending Capital"]

        for col in amount_cols_display:
             if col in df_display.columns:
                  df_display[col] = st.session_state.df_results[col].apply(lambda x: f"{currency_symbol}{format_currency(x, current_currency)}")

        if "Units Produced" in df_display.columns:
            df_display["Units Produced"] = st.session_state.df_results["Units Produced"].apply(format_units)
        if "Profit Margin (%)" in df_display.columns:
            df_display["Profit Margin (%)"] = st.session_state.df_results["Profit Margin (%)"].apply(lambda x: f"{x:.2f}%")

        display_cols_order = ["Month", "Starting Capital", "Fixed Costs", "Remaining for Prod.",
                              "Units Produced", "Revenue", "Profit Margin (%)", "Profit (JS Logic)", #"Profit (Standard)",
                              "Diseconomies Cost", "Capital Injection", "Ending Capital"]
        display_cols_final = [col for col in display_cols_order if col in df_display.columns]
        st.dataframe(df_display[display_cols_final], hide_index=True, use_container_width=True)


        # --- Display Chart ---
        st.header("Financial Projections Chart")
        # (Chart code remains the same, using st.session_state.df_results)
        chart_df = st.session_state.df_results.copy() # Use the latest calculated data

        # Prepare data for chart (apply currency conversion if needed)
        if current_currency == 'USD':
            amount_cols_chart = ["Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
                                "Profit (JS Logic)", "Profit (Standard)", "Diseconomies Cost",
                                "Capital Injection", "Ending Capital"]
            for col in amount_cols_chart:
                if col in chart_df.columns and INR_TO_USD != 0: # Check for column and valid divisor
                    chart_df[col] = chart_df[col] / INR_TO_USD

        fig = go.Figure()
        # Add traces only if columns exist
        if "Starting Capital" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Starting Capital"], mode='lines', name='Starting Capital'))
        if "Fixed Costs" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Fixed Costs"], mode='lines', name='Fixed Costs'))
        if "Revenue" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Revenue"], mode='lines', name='Revenue'))
        if "Profit (JS Logic)" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Profit (JS Logic)"], mode='lines', name='Profit (JS Logic)', line=dict(dash='dot')))
        if "Diseconomies Cost" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Diseconomies Cost"], mode='lines', name='Diseconomies Cost'))
        if "Capital Injection" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Capital Injection"], mode='lines', name='Capital Injection'))
        if "Ending Capital" in chart_df.columns: fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Ending Capital"], mode='lines', name='Ending Capital', line=dict(width=3)))


        fig.update_layout(
            title="Monthly Financial Trends",
            xaxis_title="Month",
            yaxis_title=f"Amount ({current_currency})",
            legend_title="Metrics",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Show only if calculation hasn't run or failed
    if valid_inputs:
         st.info("Click 'Generate Projections' in the sidebar to start.")
    else:
         st.warning("Please correct the errors in the sidebar inputs.")

# --- Optional: Display Raw Edits ---
# with st.expander("Show User Edits"):
#     st.write(st.session_state.user_edits)
# st.write("Editor Key:", st.session_state.editor_key) # Debug editor key
# st.write("Last Params:", st.session_state.last_params) # Debug params
# st.write("Current Params:", current_params) # Debug params
