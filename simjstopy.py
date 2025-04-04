import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import ast # For safely evaluating list inputs

# --- Constants ---
INR_TO_USD = 85.0  # Use float for division

# --- Helper Functions ---

def format_currency(amount, currency):
    """Formats currency values with mn/bn suffixes based on selected currency."""
    if currency == 'USD':
        amount /= INR_TO_USD
    
    abs_amount = abs(amount)
    sign = "-" if amount < 0 else ""

    if abs_amount >= 1e9:
        return f"{sign}{(abs_amount / 1e9):.1f} bn"
    elif abs_amount >= 1e6:
        return f"{sign}{(abs_amount / 1e6):.1f} mn"
    else:
        # Use f-string formatting for consistent decimal places
        return f"{sign}{amount:,.2f}" 

def format_units(units):
    """Formats unit counts with mn/bn suffixes."""
    abs_units = abs(units)
    sign = "-" if units < 0 else "" # Should not be negative, but good practice

    if abs_units >= 1e9:
        return f"{sign}{(abs_units / 1e9):.1f} bn"
    elif abs_units >= 1e6:
        return f"{sign}{(abs_units / 1e6):.1f} mn"
    else:
        # Format as integer if whole number, else with decimals if needed
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

# --- Streamlit App Layout ---

st.set_page_config(layout="wide") # Use wide layout for better table/chart display
st.title("Financial Projection Tool")

# --- Input Form ---
# Use columns for side-by-side layout similar to the original
col1, col2 = st.columns(2)

with col1:
    st.header("Business Parameters")
    initial_capital = st.number_input("Initial Capital (₹):", min_value=0.0, value=200000000.0, step=1000000.0, format="%.2f")
    fixed_costs_initial = st.number_input("Initial Fixed Costs (₹):", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
    variable_cost_per_unit = st.number_input("Variable Cost Per Unit (₹):", min_value=0.0, value=13.0, step=0.5, format="%.2f")
    selling_price_per_unit = st.number_input("Selling Price Per Unit (₹):", min_value=0.0, value=18.0, step=0.5, format="%.2f")
    # Note: Original JS applies this to the theoretical profit per unit for margin display, not the core profit calc. Replicating that.
    profit_growth_rate_input = st.number_input("Profit Growth Rate (% per month):", value=-1.0, step=0.1, format="%.2f", help="Affects the theoretical 'Profit Margin (%)' display calculation month-over-month.")
    
    ci_months_str = st.text_input("Capital Injection Months (e.g., [9,18,27]):", value="[9,18,27]")
    ci_amounts_str = st.text_input("Capital Injection Amounts (₹) (e.g., [200000000,200000000,-99999]):", value="[200000000,200000000,-99999]")

with col2:
    st.header("Growth & Constraints")
    fixed_cost_growth_rate_input = st.number_input("Fixed Cost Growth Rate (% per month):", value=100.0, step=1.0, format="%.2f")
    fixed_cost_cap = st.number_input("Fixed Cost Cap (₹):", min_value=0.0, value=999999999.0, step=10000.0, format="%.2f")
    diseconomies_of_scale_input = st.number_input("Diseconomies of Scale (% of Revenue):", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    months = st.number_input("Number of Months:", min_value=1, value=36, step=1)

# --- Currency Toggle ---
st.markdown("---") # Separator
current_currency = st.radio("Display Currency:", ("INR", "USD"), index=0, horizontal=True)
st.markdown("---")

# --- Input Validation and Parsing ---
# Convert percentages to decimals
profit_growth_rate = profit_growth_rate_input / 100.0
fixed_cost_growth_rate = fixed_cost_growth_rate_input / 100.0
diseconomies_of_scale = diseconomies_of_scale_input / 100.0

# Parse list inputs safely
ci_months = parse_list_input(ci_months_str, [9, 18, 27])
ci_amounts = parse_list_input(ci_amounts_str, [200000000, 200000000, -99999])

# Validate list lengths
if len(ci_months) != len(ci_amounts):
    st.error("Error: The number of 'Capital Injection Months' must match the number of 'Capital Injection Amounts'. Please check your inputs.")
    st.stop() # Halt execution if lists don't match

# Create a dictionary for quick lookup of capital injections
capital_injection_map = dict(zip(ci_months, ci_amounts))

# --- Core Calculation Logic ---
if variable_cost_per_unit >= selling_price_per_unit and profit_growth_rate <= 0:
     st.warning("Warning: Selling Price per Unit is not greater than Variable Cost Per Unit, and Profit Growth Rate is not positive. Profitability may not be achieved.")
elif variable_cost_per_unit >= selling_price_per_unit :
     st.warning("Warning: Selling Price per Unit is not greater than Variable Cost Per Unit.")


data = []
starting_capital = initial_capital
current_fixed_costs = fixed_costs_initial
# Initial profit per unit for margin calculation (this value changes based on profit_growth_rate)
profit_per_unit_for_margin = selling_price_per_unit - variable_cost_per_unit

# Check for division by zero if selling price is zero
if selling_price_per_unit == 0:
    st.error("Error: Selling Price Per Unit cannot be zero.")
    st.stop()

for month in range(1, months + 1):
    # 1. Calculate funds available after fixed costs
    remaining_for_production = starting_capital - current_fixed_costs

    # 2. Calculate units produced (avoid division by zero, handle negative remaining funds)
    units_produced = 0
    if remaining_for_production > 0 and variable_cost_per_unit > 0:
        units_produced = math.floor(remaining_for_production / variable_cost_per_unit)
    elif remaining_for_production > 0 and variable_cost_per_unit == 0:
         # If variable cost is zero, can theoretically produce infinite units
         # Handle this case based on desired business logic. Assume 0 for now to avoid infinity.
         # Or potentially set a max production capacity if that input existed.
         # For safety and matching JS where this wasn't explicitly handled:
         units_produced = 0 # Or perhaps calculate based on revenue potential vs remaining capital? Reverting to 0 if VC=0.
         st.warning(f"Month {month}: Variable cost is zero, setting units produced to 0.")


    # 3. Calculate Revenue
    revenue = units_produced * selling_price_per_unit

    # 4. Calculate Diseconomies Cost
    diseconomies_cost = revenue * diseconomies_of_scale

    # 5. Calculate Profit (using the EXACT logic from the original JS)
    # Original JS Profit = revenue - (remainingForProduction + currentFixedCosts + diseconomiesCost)
    # Which simplifies to: revenue - startingCapital - diseconomiesCost
    # *** NOTE: This differs from standard accounting profit: Revenue - FixedCosts - VariableCosts - DiseconomiesCost ***
    # To replicate the JS exactly:
    profit_js_logic = revenue - (remaining_for_production + current_fixed_costs + diseconomies_cost)
    # Let's also calculate standard profit for comparison/potential future use
    variable_costs_total = units_produced * variable_cost_per_unit
    profit_standard = revenue - current_fixed_costs - variable_costs_total - diseconomies_cost
    
    # Using the JS logic profit for ending capital calculation as per original code:
    profit_for_month = profit_js_logic

    # 6. Calculate Profit Margin Percentage (based on the evolving profit_per_unit_for_margin)
    # Avoid division by zero if selling_price_per_unit becomes zero (shouldn't happen with input validation but safe)
    profit_margin_percentage = 0.0
    if selling_price_per_unit != 0:
       # Profit margin uses the theoretical profit per unit, which grows/shrinks
       profit_margin_percentage = (profit_per_unit_for_margin / selling_price_per_unit) * 100


    # 7. Check for Capital Injection
    capital_injection = capital_injection_map.get(month, 0) # Get injection amount or 0 if month not in map

    # 8. Calculate Ending Capital
    ending_capital = starting_capital + profit_for_month + capital_injection

    # Store results for this month
    month_data = {
        "Month": month,
        "Starting Capital": starting_capital,
        "Fixed Costs": current_fixed_costs,
        "Remaining for Prod.": remaining_for_production if remaining_for_production > 0 else 0, # Show 0 if negative
        "Units Produced": units_produced,
        "Revenue": revenue,
        "Profit Margin (%)": profit_margin_percentage, # Theoretical margin based on per-unit price/cost
        "Profit (JS Logic)": profit_for_month, # Profit calculated as per original JS
        "Profit (Standard)": profit_standard, # Standard accounting profit
        "Diseconomies Cost": diseconomies_cost,
        "Capital Injection": capital_injection,
        "Ending Capital": ending_capital,
    }
    data.append(month_data)

    # --- Update for Next Month ---
    starting_capital = ending_capital
    current_fixed_costs = min(current_fixed_costs * (1 + fixed_cost_growth_rate), fixed_cost_cap)
    # Update the theoretical profit per unit used for margin display
    profit_per_unit_for_margin *= (1 + profit_growth_rate)
    # Selling price per unit remains constant in the original logic unless profit_per_unit logic implicitly changes it.
    # Based on JS, selling_price_per_unit is static input. profit_per_unit calculation affects ONLY the margin display.


# --- Create DataFrame ---
df = pd.DataFrame(data)

# --- Display Table ---
st.header("Financial Projections Table")

# Create a copy for display formatting
df_display = df.copy()

# Apply formatting based on selected currency
currency_symbol = "₹" if current_currency == "INR" else "$"
amount_cols = ["Starting Capital", "Fixed Costs", "Remaining for Prod.", "Revenue",
               "Profit (JS Logic)", "Profit (Standard)", "Diseconomies Cost", "Capital Injection", "Ending Capital"]

for col in amount_cols:
    df_display[col] = df[col].apply(lambda x: f"{currency_symbol}{format_currency(x, current_currency)}")

df_display["Units Produced"] = df["Units Produced"].apply(format_units)
df_display["Profit Margin (%)"] = df["Profit Margin (%)"].apply(lambda x: f"{x:.2f}%")

# Select columns to display (including the standard profit for info)
display_cols = ["Month", "Starting Capital", "Fixed Costs", "Remaining for Prod.",
                "Units Produced", "Revenue", "Profit Margin (%)", "Profit (JS Logic)", "Profit (Standard)",
                "Diseconomies Cost", "Capital Injection", "Ending Capital"]
st.dataframe(df_display[display_cols], hide_index=True) # Use st.dataframe for better interactivity


# --- Display Chart ---
st.header("Financial Projections Chart")

# Prepare data for chart (apply currency conversion if needed)
chart_df = df.copy()
if current_currency == 'USD':
    for col in amount_cols:
        chart_df[col] = chart_df[col] / INR_TO_USD

fig = go.Figure()

# Add traces for key metrics
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Starting Capital"], mode='lines', name='Starting Capital'))
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Fixed Costs"], mode='lines', name='Fixed Costs'))
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Revenue"], mode='lines', name='Revenue'))
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Profit (JS Logic)"], mode='lines', name='Profit (JS Logic)', line=dict(dash='dot'))) # Use JS profit for main chart line as per original
# fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Profit (Standard)"], mode='lines', name='Profit (Standard)', line=dict(dash='dash'))) # Optional: show standard profit too
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Diseconomies Cost"], mode='lines', name='Diseconomies Cost'))
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Capital Injection"], mode='lines', name='Capital Injection'))
fig.add_trace(go.Scatter(x=chart_df["Month"], y=chart_df["Ending Capital"], mode='lines', name='Ending Capital', line=dict(width=3))) # Thicker line for emphasis

# Update layout
fig.update_layout(
    title="Monthly Financial Trends",
    xaxis_title="Month",
    yaxis_title=f"Amount ({current_currency})",
    legend_title="Metrics",
    hovermode="x unified" # Improves tooltip display
)

st.plotly_chart(fig, use_container_width=True)

# --- Optional: Display Raw Data ---
# with st.expander("Show Raw Calculated Data (INR)"):
#     st.dataframe(df)
