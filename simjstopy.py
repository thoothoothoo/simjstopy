import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import ast
from copy import deepcopy # To compare dataframes without modifying originals

# --- Constants ---
INR_TO_USD = 85.0

# --- Helper Functions ---

def format_currency(amount, currency):
    """Formats currency values with mn/bn suffixes based on selected currency."""
    display_amount = amount
    if currency == 'USD':
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
    except (ValueError, SyntaxError, TypeError):
        st.warning(f"Could not parse input '{input_string}' as a list. Using default: {default_value}")
        return default_value

# --- Core Calculation Function ---
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
    profit_per_unit_for_margin = selling_price_per_unit - variable_cost_per_unit

    # Validation moved outside loop for efficiency
    if selling_price_per_unit <= 0:
         # We need a price to calculate revenue and margin %
         st.error("Error: Selling Price Per Unit must be positive.")
         return pd.DataFrame() # Return empty DataFrame on critical error


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
        if remaining_for_production > 0 and variable_cost_per_unit > 0:
            units_produced = math.floor(remaining_for_production / variable_cost_per_unit)
        elif remaining_for_production > 0 and variable_cost_per_unit <= 0: # Handle zero/negative VC
             units_produced = 0 # Avoid infinite production / division issues
             # Could add a warning if desired

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
        prev_fixed_costs = effective_fixed_costs
        current_fixed_costs = min(prev_fixed_costs * (1 + fixed_cost_growth_rate), fixed_cost_cap)
        # Update the theoretical profit per unit for margin display
        profit_per_unit_for_margin *= (1 + profit_growth_rate)

    return pd.DataFrame(data)

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("Financial Projection Tool (Editable)")

# --- Initialize Session State ---
if 'df_results' not in st.session_state:
    st.session_state.df_results = pd.DataFrame() # Holds the main data
if 'user_edits' not in st.session_state:
    st.session_state.user_edits = {} # Holds {(month, col): value} overrides
if 'last_data_editor_state' not in st.session_state:
     st.session_state.last_data_editor_state = {} # To detect changes in data_editor


# --- Input Form ---
with st.sidebar: # Move inputs to sidebar for more space
    st.header("Initial Parameters")
    initial_capital = st.number_input("Initial Capital (₹):", min_value=0.0, value=200000000.0, step=1000000.0, format="%.2f")
    fixed_costs_initial = st.number_input("Initial Fixed Costs (₹):", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    variable_cost_per_unit = st.number_input("Variable Cost Per Unit (₹):", min_value=0.0, value=13.0, step=0.5, format="%.2f")
    selling_price_per_unit = st.number_input("Selling Price Per Unit (₹):", min_value=0.0, value=18.0, step=0.5, format="%.2f")
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

# Validate list lengths
valid_inputs = True
if len(ci_months) != len(ci_amounts):
    st.sidebar.error("Error: Injection months and amounts lists must have the same length.")
    valid_inputs = False
    st.stop() # Prevent calculation with mismatched lists

# Generate Projections Button Clicked
if generate_button and valid_inputs:
    st.session_state.user_edits = {} # Clear previous edits when generating anew
    st.session_state.df_results = calculate_projections(
        initial_capital, fixed_costs_initial, variable_cost_per_unit,
        selling_price_per_unit, profit_growth_rate_input, ci_months, ci_amounts,
        fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
        months, user_edits={} # Start fresh
    )
    st.session_state.last_data_editor_state = {} # Reset editor state tracking
    st.rerun() # Rerun to update the display immediately after calculation

# Clear Edits Button Clicked
if clear_edits_button and valid_inputs:
    if st.session_state.user_edits: # Only recalculate if there were edits to clear
        st.session_state.user_edits = {}
        st.session_state.df_results = calculate_projections(
            initial_capital, fixed_costs_initial, variable_cost_per_unit,
            selling_price_per_unit, profit_growth_rate_input, ci_months, ci_amounts,
            fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
            months, user_edits={} # Recalculate without edits
        )
        st.session_state.last_data_editor_state = {} # Reset editor state tracking
        st.sidebar.success("Edits cleared and projections recalculated.")
        st.rerun()
    else:
        st.sidebar.info("No user edits to clear.")


# --- Display Results ---
if not st.session_state.df_results.empty:

    st.header("Editable Projections Table")
    st.caption("You can edit cells in the 'Fixed Costs' and 'Capital Injection' columns. Changes will recalculate subsequent months.")

    # Prepare dataframe for editing (use raw numbers)
    df_editable = st.session_state.df_results.copy()

    # Define which columns are editable
    # Make all columns read-only by default
    column_config = {col: st.column_config.NumberColumn(disabled=True, format="%.2f") for col in df_editable.columns if col != "Month"}
    # Specifically enable editing for Fixed Costs and Capital Injection
    column_config["Fixed Costs"] = st.column_config.NumberColumn(
        label="Fixed Costs (Editable)", # Custom label
        min_value=0, # Example validation: Fixed costs cannot be negative
        format="%.2f", # Ensure proper formatting display in edit mode
        disabled=False
        )
    column_config["Capital Injection"] = st.column_config.NumberColumn(
        label="Capital Injection (Editable)",
        format="%.2f",
        disabled=False
        )
    # Configure Month column separately if needed (usually just displayed)
    column_config["Month"] = st.column_config.NumberColumn(disabled=True)
    # Configure other specific formatting if needed
    column_config["Units Produced"] = st.column_config.NumberColumn(disabled=True, format="%d") # Integer format
    column_config["Profit Margin (%)"] = st.column_config.NumberColumn(disabled=True, format="%.2f%%")


    # Store the state of the dataframe *before* showing the editor
    df_before_edit = df_editable.copy()

    # Display the data editor
    edited_df = st.data_editor(
        df_editable,
        column_config=column_config,
        key="data_editor", # Assign a key to persist editor state correctly
        hide_index=True,
        num_rows="dynamic", # Allows adding/deleting rows, maybe disable if not desired: use "fixed"
        use_container_width=True
    )

    # --- Detect and Process Edits ---
    # Check if the dataframe returned by the editor is different from the one fed into it
    if not edited_df.equals(df_before_edit):
        # Find changes (more robust comparison needed for production)
        # This is a simple diff; more complex logic might be needed for row add/delete
        try:
            diff = edited_df.compare(df_before_edit)
            new_edits_detected = False
            for idx in diff.index:
                month = edited_df.loc[idx, "Month"] # Get the month for the changed row
                changed_cols = diff.loc[idx].dropna().index.get_level_values(0).unique()

                for col in changed_cols:
                    # Only process edits in allowed columns
                    if col in ["Fixed Costs", "Capital Injection"]:
                        new_value = edited_df.loc[idx, col]
                        # Basic Validation (Example: ensure non-negative Fixed Costs)
                        if col == "Fixed Costs" and new_value < 0:
                            st.warning(f"Invalid Edit: Fixed Costs for Month {month} cannot be negative. Reverting change.")
                            # Revert the change in the edited_df before recalculating
                            # This part is tricky with data_editor's direct modification.
                            # A cleaner way is to ignore the invalid edit when updating user_edits
                            continue # Skip this invalid edit


                        # Check if this specific edit is actually new or different
                        edit_key = (month, col)
                        if edit_key not in st.session_state.user_edits or st.session_state.user_edits[edit_key] != new_value:
                            st.session_state.user_edits[edit_key] = new_value
                            new_edits_detected = True
                            # st.toast(f"Change detected: Month {month}, {col} = {new_value}") # Optional feedback

            # If any valid new edit was detected, recalculate
            if new_edits_detected and valid_inputs:
                 st.session_state.df_results = calculate_projections(
                    initial_capital, fixed_costs_initial, variable_cost_per_unit,
                    selling_price_per_unit, profit_growth_rate_input, ci_months, ci_amounts,
                    fixed_cost_growth_rate_input, fixed_cost_cap, diseconomies_of_scale_input,
                    months, st.session_state.user_edits # Pass current edits
                )
                 st.rerun() # Rerun immediately to show recalculated data in editor/chart

        except Exception as e:
            st.error(f"Error processing edits: {e}")
            # Potentially reset or handle error state


    # --- Display Formatted Table (Read-Only) ---
    st.header("Formatted Projections (Read-Only)")
    df_display = st.session_state.df_results.copy()
    currency_symbol = "₹" if current_currency == "INR" else "$"
    amount_cols_display = ["Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
                           "Profit (JS Logic)", "Profit (Standard)", "Diseconomies Cost", "Capital Injection", "Ending Capital"]

    for col in amount_cols_display:
         # Ensure column exists before formatting
         if col in df_display.columns:
              df_display[col] = st.session_state.df_results[col].apply(lambda x: f"{currency_symbol}{format_currency(x, current_currency)}")

    if "Units Produced" in df_display.columns:
        df_display["Units Produced"] = st.session_state.df_results["Units Produced"].apply(format_units)
    if "Profit Margin (%)" in df_display.columns:
        df_display["Profit Margin (%)"] = st.session_state.df_results["Profit Margin (%)"].apply(lambda x: f"{x:.2f}%")

    # Select and order columns for display
    display_cols_order = ["Month", "Starting Capital", "Fixed Costs", "Remaining for Prod.",
                          "Units Produced", "Revenue", "Profit Margin (%)", "Profit (JS Logic)", #"Profit (Standard)",
                          "Diseconomies Cost", "Capital Injection", "Ending Capital"]
    # Filter out columns that might not exist if calculation failed
    display_cols_final = [col for col in display_cols_order if col in df_display.columns]
    st.dataframe(df_display[display_cols_final], hide_index=True, use_container_width=True)


    # --- Display Chart ---
    st.header("Financial Projections Chart")
    chart_df = st.session_state.df_results.copy() # Use the latest calculated data

    # Prepare data for chart (apply currency conversion if needed)
    if current_currency == 'USD':
        amount_cols_chart = ["Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
                             "Profit (JS Logic)", "Profit (Standard)", "Diseconomies Cost",
                             "Capital Injection", "Ending Capital"]
        for col in amount_cols_chart:
             if col in chart_df.columns:
                chart_df[col] = chart_df[col] / INR_TO_USD

    fig = go.Figure()
    # Add traces only if columns exist
    if "Starting Capital" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Starting Capital"], mode='lines', name='Starting Capital'))
    if "Fixed Costs" in chart_df.columns:
        fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Fixed Costs"], mode='lines', name='Fixed Costs'))
    # ... (add similar checks for other columns) ...
    if "Revenue" in chart_df.columns:
      fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Revenue"], mode='lines', name='Revenue'))
    if "Profit (JS Logic)" in chart_df.columns:
      fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Profit (JS Logic)"], mode='lines', name='Profit (JS Logic)', line=dict(dash='dot')))
    if "Diseconomies Cost" in chart_df.columns:
      fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Diseconomies Cost"], mode='lines', name='Diseconomies Cost'))
    if "Capital Injection" in chart_df.columns:
      fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Capital Injection"], mode='lines', name='Capital Injection'))
    if "Ending Capital" in chart_df.columns:
      fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Ending Capital"], mode='lines', name='Ending Capital', line=dict(width=3)))


    fig.update_layout(
        title="Monthly Financial Trends",
        xaxis_title="Month",
        yaxis_title=f"Amount ({current_currency})",
        legend_title="Metrics",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Click 'Generate Projections' in the sidebar to start.")

# --- Optional: Display Raw Edits ---
# with st.expander("Show User Edits"):
#     st.write(st.session_state.user_edits)
