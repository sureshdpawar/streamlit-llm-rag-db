import snowflake.connector
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import prompts

st.set_page_config(layout="wide")

# Variables
sf_db = st.secrets["sf_database"]
sf_schema = st.secrets["sf_schema"]
tick_list = ['BRK.A','AAPL','PG','JNJ','MA','MCO','VZ','KO','AXP', 'BAC']
fin_statement_list = ['income_statement','balance_sheet','cash_flow_statement']
year_cutoff = 20 # year cutoff for financial statement plotting

# establish snowpark connection
#conn = st.connection("snowflake")

# Assuming your secrets.toml is correctly set up as shown in your question
conn_info = st.secrets["connections"]["snowpark"]

# Establish Snowflake connection
conn = snowflake.connector.connect(
    user=conn_info["user"],
    password=conn_info["password"],
    account=conn_info["account"],
    warehouse=conn_info["warehouse"],
    database=conn_info["database"],
    schema=conn_info["schema"],
    client_session_keep_alive=conn_info.get("client_session_keep_alive", False),
)


# Reset the connection before using it if it isn't healthy
# Assuming conn is your SnowflakeConnection object
cursor = conn.cursor()
try:
    cursor.execute("SELECT 1")
    # Fetch results, if necessary
    one_row = cursor.fetchone()
    print(one_row)
finally:
    cursor.close()


#@st.cache_data
def pull_financials(database, schema, statement, ticker):
    """
    query to pull financial data from snowflake based on database, schema, statement and ticker
    """
    cursor = conn.cursor()
    try:
        # Prepare the SQL query string
        query = f"SELECT * FROM {database}.{schema}.{statement} WHERE ticker = '{ticker}' ORDER BY year DESC"
        # Execute the query
        cursor.execute(query)

        # Fetch all rows from the query result
        rows = cursor.fetchall()

        # Assuming the cursor description attribute can be used to retrieve column names
        column_names = [desc[0] for desc in cursor.description]

        # Create a DataFrame from the query result
        df = pd.DataFrame(rows, columns=column_names)

        return df
    finally:
        # Ensure the cursor is closed after operation
        cursor.close()

# metrics for kpi cards
#@st.cache_data
def kpi_recent(df, metric, periods=2, unit=1000000000):
    """
    filters a financial statement dataframe down to the most recent periods
    df is the financial statement. Metric is the column to be used.
    """
    return df.sort_values('year',ascending=False).head(periods)[metric]/unit

def plot_financials(df, x, y, x_cutoff, title):
    """"
    helper to plot the altair financial charts
    """
    return st.altair_chart(alt.Chart(df.head(x_cutoff)).mark_bar().encode(
        x=x,
        y=y
        ).properties(title=title)
    ) 

# adding this to test out caching
#st.cache_data(ttl=86400)
def fs_chain(str_input):
    """
    performs qa capability for a question using sql vector db store
    the prompts.fs_chain is used but with caching
    """
    output = prompts.fs_chain(str_input)
    return output

# adding this to test out caching
#st.cache_data(ttl=86400)
def sf_query(str_input):
    """
    performs snowflake query with caching
    """
    #data = conn.query(str_input)

    cursor = conn.cursor()
    try:
        # Prepare the SQL query string safely with placeholders for variables
        query = str_input
        print(f"Preparing to execute query: {query}")  # Debugging: Print the query to be executed

        # Execute the query with proper parameter substitution to avoid SQL injection
        cursor.execute(query)  # Pass parameters as a list or tuple

        # Fetch all rows from the query result
        rows = cursor.fetchall()
        print(f"Fetched {len(rows)} rows.")  # Debugging: Print the number of rows fetched

        if rows:  # Check if any rows were fetched
            # Retrieve column names from the cursor description
            column_names = [desc[0] for desc in cursor.description]

            # Create a DataFrame from the query result
            df = pd.DataFrame(rows, columns=column_names)

            return df
        else:
            print("No data fetched.")
            return pd.DataFrame()  # Return an empty DataFrame if no rows were fetched
    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging: Print any error that occurs
        raise  # Optionally re-raise the exception for further handling
    finally:
        cursor.close()  # Ensure the cursor is always closed

    #print("data", data)
    #return data

# create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Financial Statement Questions :dollar:",
    "Financial Data Exploration :chart_with_upwards_trend:",
    "Shareholder Letter Questions :memo:",
    "Additional Details :notebook:"]
    )

with st.sidebar:
    st.markdown("""
      Developed by Suresh Pawar
    """)

with tab1:
    st.markdown("""
    # Financial Statement Questions :dollar:
    ### Leverage LLMs to translate natural language questions related to financial statements and turn those into direct Snowflake queries
    Data is stored and queried directly from income statement, balance sheet, and cash flow statement in Snowflake

    **Example questions to ask:**

    - What was Proctor and Gamble's net income from 2010 through 2020?
    - What was Apple's debt to equity ratio for the last 5 years?
    - Rank the companies in descending order based on their net income in 2022. Include the ticker and net income value
    - What has been the average for total assets and total liabilities for each company over the last 3 years? List the top 3
    """
    )

    str_input = st.text_input(label='What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        print(str_input)
        with st.spinner('Looking up your question in Snowflake now...'):
            try:
                output = fs_chain(str_input)
                print("output", output)
                #st.write(output)
                try:
                    # if the output doesn't work we will try one additional attempt to fix it
                    query_result = sf_query(output['result'])
                    print("query result", query_result)
                    if len(query_result) > 1:
                        st.write(query_result)
                        st.write(output)
                except:
                    st.write("The first attempt didn't pull what you were needing. Trying again...")
                    output = fs_chain(f'You need to fix the code but ONLY produce SQL code output. If the question is complex, consider using one or more CTE. Examine the DDL statements and answer this question: {output}')
                    st.write(sf_query(output['result']))
                    st.write(output)
            except:
                st.write("Please try to improve your question. Note this tab is for financial statement questions. Use Tab 3 to ask from shareholder letters. Also, only a handful of companies are available, which you can see on the side bar.")
                st.write(f"Final errored query used:")
                st.write(output)



