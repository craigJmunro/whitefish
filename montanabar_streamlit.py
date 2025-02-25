import streamlit as st
import numpy as np
import numpy_financial as npf  # For IRR calculation
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
#     MODEL CONSTANTS
# ----------------------------
# We'll compute a custom variable cost ratio from the food/drink percentages.
DEFAULT_SEASONALITY_OPTIONS = {
    "Quiet": 0.8,
    "Normal": 1.0,
    "Busy": 1.2,
    "Very Busy": 1.4
}
# Suggested defaults for Whitefish, MT (first 12 months: March to February)
default_seasonality_defaults = {
    "March": "Quiet",
    "April": "Quiet",
    "May": "Normal",
    "June": "Busy",
    "July": "Very Busy",
    "August": "Very Busy",
    "September": "Very Busy",
    "October": "Busy",
    "November": "Normal",
    "December": "Normal",
    "January": "Busy",
    "February": "Normal"
}
ANNUAL_DISCOUNT_RATE = 0.10

# ----------------------------
#     HELPER FUNCTIONS
# ----------------------------
def monthly_discount_rate(annual_rate):
    return (1 + annual_rate)**(1/12) - 1

# (Loan functions remain for completeness, but we are in a 100% equity scenario.)
def calculate_loan_payment(loan_amount, annual_interest, term_months):
    monthly_rate = annual_interest/12
    if monthly_rate == 0 or loan_amount == 0:
        return 0
    return loan_amount * monthly_rate / (1 - (1+monthly_rate)**(-term_months))

def monthly_loan_balance(previous_balance, monthly_payment, annual_interest):
    monthly_rate = annual_interest/12
    interest_paid = previous_balance * monthly_rate
    principal_paid = monthly_payment - interest_paid
    new_balance = previous_balance - principal_paid
    return new_balance, interest_paid, principal_paid

def discounted_value(amount, discount_rate, month):
    return amount / ((1+discount_rate)**month)

def compute_irr(initial_investment, monthly_net_cfs):
    cash_flows = [-initial_investment] + monthly_net_cfs
    irr_monthly = npf.irr(cash_flows)
    if irr_monthly is None or np.isnan(irr_monthly):
        return None
    irr_annual = (1+irr_monthly)**12 - 1
    return irr_annual

# ----------------------------
# SINGLE SIMULATION (100% Equity)
# ----------------------------
def run_monthly_simulation(
    daily_cust_mean,
    spend_mean,
    months,
    days_in_month_list,        # List of actual days in each month
    include_seasonality,
    seasonality_factors,       # List of multipliers for months 1-12 (from dropdowns)
    baseline_monthly_growth,
    initial_investment,        # Effective initial investment (equals total investment)
    fixed_cost_non_wage,       # Monthly fixed costs (non-wage)
    wage_per_hour,
    hours_per_day,
    staff_per_day,
    var_cost_ratio,            # Computed variable cost ratio from food & drink percentages
    owner_share                # For 100% equity, owner_share = 1.0
):
    # Deterministic daily revenue:
    daily_revenue = daily_cust_mean * spend_mean

    if not include_seasonality:
        seasonality_factors = [1.0] * 12

    mdr = monthly_discount_rate(ANNUAL_DISCOUNT_RATE)

    # No loan financing:
    monthly_payment = 0  
    loan_balance = 0

    cumulative_cf = 0.0
    cumulative_disc_cf = 0.0
    payback_month = None
    monthly_data = []
    monthly_net_cfs = []  # For IRR calculation

    for m in range(1, months+1):
        days_current = days_in_month_list[m-1]
        if m == 1:
            revenue_month = daily_revenue * days_current
        else:
            daily_revenue *= (1 + baseline_monthly_growth)
            revenue_month = daily_revenue * days_current

        if include_seasonality and m <= len(seasonality_factors):
            season_factor = seasonality_factors[m-1]
        else:
            season_factor = 1.0
        revenue_month *= season_factor

        var_costs = revenue_month * var_cost_ratio

        # Calculate wage cost based on actual days in the month:
        wage_cost = wage_per_hour * hours_per_day * staff_per_day * days_current
        fixed_costs = fixed_cost_non_wage + wage_cost

        business_net_cf = revenue_month - var_costs - fixed_costs  # No loan payments
        net_cf = business_net_cf * owner_share

        monthly_net_cfs.append(net_cf)

        disc_cf = discounted_value(net_cf, mdr, m)
        cumulative_cf += net_cf
        cumulative_disc_cf += disc_cf

        if payback_month is None and cumulative_cf >= initial_investment:
            payback_month = m

        monthly_data.append({
            "Month": m,
            "Revenue": revenue_month,
            "Variable Costs": var_costs,
            "Fixed Costs": fixed_costs,
            "Net CF": net_cf,
            "Cumulative CF": cumulative_cf,
            "Discounted CF": disc_cf,
            "Cumulative Discounted CF": cumulative_disc_cf
        })

    npv = cumulative_disc_cf - initial_investment
    irr_annual = compute_irr(initial_investment, monthly_net_cfs)
    return {"monthly_data": monthly_data, "NPV": npv, "IRR": irr_annual, "payback_month": payback_month}

# ----------------------------
# STREAMLIT APP
# ----------------------------
def main():
    st.title("Whitefish Bar Investment Model (100% Equity)")
    st.markdown("""
    This app models the financial performance for a bar/restaurant in Whitefish, Montana,
    assuming all required capital is raised through equity.
    Adjust the inputs below and click **"Run Simulation"** to update the results dynamically.
    """)

    # INITIAL INVESTMENT COSTS (Freehold)
    st.subheader("Initial Investment Costs")
    st.markdown("Enter the cost to acquire the freehold and remodel the bar/restaurant.")
    include_freehold = st.checkbox("Include Freehold Cost?", value=True, key="include_freehold")
    include_renovation = st.checkbox("Include Renovations?", value=True, key="include_renovation")
    include_licenses = st.checkbox("Include Licenses?", value=True, key="include_licenses")
    freehold_cost = st.number_input("Freehold Cost", min_value=0, max_value=10000000, value=2700000, step=10000, key="freehold_cost")
    reno_cost = st.number_input("Renovation Cost", min_value=0, max_value=10000000, value=700000, step=10000, key="reno_cost")
    licenses_cost = st.number_input("Licenses Cost", min_value=0, max_value=10000000, value=50000, step=1000, key="licenses_cost")
    if not include_freehold:
        freehold_cost = 0
    if not include_renovation:
        reno_cost = 0
    if not include_licenses:
        licenses_cost = 0
    total_investment = freehold_cost + reno_cost + licenses_cost
    st.write(f"**Total Investment Cost:** ${total_investment:,.0f}")

    owner_share = 1.0  # 100% equity
    effective_investment = total_investment
    st.write(f"**Owner's Effective Investment:** ${effective_investment:,.0f}")

    # MONTHLY FIXED COSTS (Non-Wage)
    st.subheader("Monthly Fixed Costs (Non-Wage)")
    st.markdown("Enter the estimated monthly fixed costs for items that do not vary with the number of days (e.g., licenses, rates, electricity).")
    include_monthly_licenses = st.checkbox("Include Monthly Licenses?", value=True, key="include_monthly_licenses")
    include_rates = st.checkbox("Include Rates?", value=True, key="include_rates")
    include_electricity = st.checkbox("Include Electricity?", value=True, key="include_electricity")
    include_other = st.checkbox("Include Other Fixed Costs?", value=False, key="include_other")
    monthly_licenses = st.number_input("Monthly Licenses", min_value=0, max_value=50000, value=1000, step=100, key="monthly_licenses")
    rates = st.number_input("Rates", min_value=0, max_value=50000, value=500, step=50, key="rates")
    electricity = st.number_input("Electricity", min_value=0, max_value=50000, value=300, step=50, key="electricity")
    other_fixed = st.number_input("Other Fixed Costs", min_value=0, max_value=50000, value=0, step=50, key="other_fixed")
    fixed_cost_non_wage = 0
    if include_monthly_licenses:
        fixed_cost_non_wage += monthly_licenses
    if include_rates:
        fixed_cost_non_wage += rates
    if include_electricity:
        fixed_cost_non_wage += electricity
    if include_other:
        fixed_cost_non_wage += other_fixed
    st.write(f"**Total Non-Wage Fixed Costs:** ${fixed_cost_non_wage:,.0f}")

    # STAFF COST ASSUMPTIONS
    st.subheader("Staff Cost Assumptions")
    wage_per_hour = st.number_input("Wage per Hour ($)", min_value=0.0, max_value=100.0, value=15.0, step=0.5, key="wage_per_hour")
    hours_per_day = st.number_input("Hours per Day", min_value=0, max_value=24, value=8, step=1, key="hours_per_day")
    staff_per_day = st.number_input("Number of Staff per Day", min_value=0, max_value=50, value=3, step=1, key="staff_per_day")

    # OPERATING ASSUMPTIONS
    st.subheader("Operating Assumptions")
    include_seasonality = st.checkbox("Include Seasonality for the First Year?", value=True, key="include_seasonality")
    if include_seasonality:
        st.markdown("For each month of the first year, select the expected customer traffic level:")
        seasonality_factors = []
        season_dates = pd.date_range("2025-03-01", periods=12, freq="MS")
        for date in season_dates:
            month_name = date.strftime("%B")
            default_choice = default_seasonality_defaults.get(month_name, "Normal")
            choice = st.selectbox(f"{month_name}", list(DEFAULT_SEASONALITY_OPTIONS.keys()),
                                  index=list(DEFAULT_SEASONALITY_OPTIONS.keys()).index(default_choice),
                                  key=f"season_select_{month_name}")
            seasonality_factors.append(DEFAULT_SEASONALITY_OPTIONS[choice])
    else:
        seasonality_factors = [1.0] * 12

    avg_daily_customers = st.slider("Average Daily Customers (Month 1)", 0, 500, 205, key="avg_daily_customers")
    avg_spend = st.slider("Average Spend Per Customer ($)", 0, 200, 40, key="avg_spend")
    
    # Revenue breakdown: food vs. drink percentages and cost assumptions.
    food_pct = st.slider("Food Revenue Percentage", 0, 100, 65, key="food_pct")
    drink_pct = 100 - food_pct
    food_cost_percentage = st.number_input("Food Cost Percentage (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5, key="food_cost_percentage")
    drink_cost_percentage = st.number_input("Drink Cost Percentage (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.5, key="drink_cost_percentage")
    computed_var_cost_ratio = (food_pct/100)*(food_cost_percentage/100) + (drink_pct/100)*(drink_cost_percentage/100)

    months = st.number_input("Number of Months to Project", 1, 120, 24, key="months")
    n_sims = st.number_input("Monte Carlo Simulations", 100, 10000, 1000, step=100, key="n_sims")  # Not used now, but left for compatibility if needed.
    monthly_growth = st.slider("Baseline Monthly Growth Rate (%)", 0.0, 10.0, 2.0, step=0.5, key="monthly_growth")
    baseline_growth = monthly_growth / 100.0

    # Valuation multiplier for the multiplier approach:
    multiplier = st.number_input("Valuation Multiplier", 1.0, 20.0, 5.0, step=0.5, key="multiplier")

    # Calculate Actual Days in Each Month:
    start_date = pd.to_datetime("2025-03-01")
    dates = pd.date_range(start_date, periods=months, freq='MS')
    days_in_month_list = [pd.Period(date, freq='M').days_in_month for date in dates]

    if st.button("Run Simulation", key="run_sim"):
        # Run a single simulation
        detail = run_monthly_simulation(
            daily_cust_mean=avg_daily_customers,
            spend_mean=avg_spend,
            months=months,
            days_in_month_list=days_in_month_list,
            include_seasonality=include_seasonality,
            seasonality_factors=seasonality_factors,
            baseline_monthly_growth=baseline_growth,
            initial_investment=effective_investment,
            fixed_cost_non_wage=fixed_cost_non_wage,
            wage_per_hour=wage_per_hour,
            hours_per_day=hours_per_day,
            staff_per_day=staff_per_day,
            var_cost_ratio=computed_var_cost_ratio,
            owner_share=owner_share
        )

        # Display key metrics:
        st.write(f"**NPV (Single Scenario):** ${detail['NPV']:,.2f}")
        if detail["IRR"] is not None:
            st.write(f"**Average Annual IRR:** {detail['IRR']*100:.1f}%")
        else:
            st.write("**Average Annual IRR:** Not computable")
        if detail["payback_month"]:
            payback_date = dates[int(detail["payback_month"]) - 1].strftime("%B-%y")
        else:
            payback_date = "Never"
        st.write(f"**Payback Month:** {payback_date}")

        # Multiplier valuation approach:
        n_months_for_avg = min(12, months)
        recent_months = detail["monthly_data"][-n_months_for_avg:]
        avg_monthly_net_cf = np.mean([m["Net CF"] for m in recent_months])
        annual_net_cf = avg_monthly_net_cf * 12
        business_value_multiplier = annual_net_cf * multiplier
        
        # DCF-based business value:
        business_value_dcf = effective_investment + detail["NPV"]
        st.write(f"**Estimated Total Business Value (DCF):** ${business_value_dcf:,.0f}")
        st.write(f"**Estimated Total Business Value (Multiplier):** ${business_value_multiplier:,.0f}")
        st.write(f"**Estimated Value of a 33% Stake (DCF):** ${0.33 * business_value_dcf:,.0f}")
        st.write(f"**Estimated Value of a 33% Stake (Multiplier):** ${0.33 * business_value_multiplier:,.0f}")

        # Build and display the monthly Profit & Loss / Cash Flow table:
        df = pd.DataFrame(detail["monthly_data"])
        if "Month" in df.columns:
            df = df.drop(columns=["Month"])
        date_str = dates.strftime("%B-%y")
        df.index = date_str
        df_transposed = df.transpose().round(2)
        st.subheader("Profit & Loss / Cash Flow Table (Transposed)")
        st.dataframe(df_transposed)

        # Build a monthly P&L Statement
        # Assume Depreciation = (Freehold Cost + Renovation Cost) / (20 years * 12 months)
        depreciation = (freehold_cost + reno_cost) / (20 * 12)
        pl_data = pd.DataFrame(detail["monthly_data"])
        pl_data["Gross Profit"] = pl_data["Revenue"] - pl_data["Variable Costs"]
        pl_data["EBITDA"] = pl_data["Gross Profit"] - pl_data["Fixed Costs"]  # EBITDA before depreciation
        pl_data["Depreciation"] = depreciation
        pl_data["EBIT"] = pl_data["EBITDA"] - pl_data["Depreciation"]
        pl_data["Tax"] = pl_data["EBIT"].apply(lambda x: 0.2 * x if x > 0 else 0)
        pl_data["Net Income"] = pl_data["EBIT"] - pl_data["Tax"]
        pl_columns = ["Revenue", "Variable Costs", "Gross Profit", "Fixed Costs", "EBITDA", "Depreciation", "EBIT", "Tax", "Net Income"]
        pl_table = pl_data[pl_columns].copy()
        pl_table.index = date_str
        st.subheader("Monthly P&L Statement")
        st.dataframe(pl_table)

        # Build a yearly summary (P&L)
        pl_data["Date"] = dates[:len(pl_data)]
        pl_data["Year"] = pl_data["Date"].dt.year
        yearly_pl = pl_data.groupby("Year")[pl_columns].sum().reset_index()
        st.subheader("Yearly P&L Summary")
        st.dataframe(yearly_pl)

        # 33% Stake Summary Table
        df_year = pd.DataFrame(detail["monthly_data"])
        df_year["Date"] = dates[:len(df_year)]
        df_year["Year"] = df_year["Date"].dt.year
        summary_yearly = df_year.groupby("Year")[["Revenue", "Variable Costs", "Net CF"]].sum().reset_index()
        summary_yearly = summary_yearly.round(2)
        summary_yearly["Stake Cash Received"] = summary_yearly["Net CF"] * 0.33
        summary_yearly["Stake Valuation (Multiplier)"] = summary_yearly["Net CF"] * multiplier * 0.33
        summary_yearly["Total Stake Value"] = summary_yearly["Stake Cash Received"] + summary_yearly["Stake Valuation (Multiplier)"]
        st.subheader("33% Stake Summary (Yearly)")
        st.dataframe(summary_yearly[["Year", "Stake Cash Received", "Stake Valuation (Multiplier)", "Total Stake Value"]])

if __name__ == "__main__":
    main()
