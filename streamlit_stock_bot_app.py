import snowflake.connector
import pandas as pd
import streamlit as st
import prompts
import requests

st.set_page_config(layout="wide")

# Variables for Snowflake
sf_db = st.secrets["sf_database"]
sf_schema = st.secrets["sf_schema"]
conn_info = st.secrets["connections"]["snowpark"]

# Sidebar for OpenAI Key input
with st.sidebar:
    st.markdown("## Configuration")
    open_ai_key = st.text_input("Enter your OpenAI API Key:")
    st.markdown("Developed by Suresh Pawar")

def fs_chain(str_input):
    """
    performs qa capability for a question using sql vector db store
    the prompts.fs_chain is used but with caching
    """
    output = prompts.fs_chain(str_input)
    return output

# Function to check if OpenAI key is valid (simple placeholder function)
def is_valid_key(key):
    # Placeholder for actual validation logic
    # You should replace this with actual validation against OpenAI API
    # URL for a lightweight API call (e.g., listing available models)
    url = "https://api.openai.com/v1/models"

    # Make the API request
    response = requests.get(url, headers={"Authorization": f"Bearer {key}"})

    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        print("valid key")
        return True
    else:
        print("Invalid key")
        return False
    #return len(key) > 0

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

# Assuming you have a function to pull financials and execute queries as defined earlier
# Definitions would remain the same, so not repeated here for brevity

st.markdown("""
# Generative AI Powered Stock Market Information Bot 
### 
Leveraging Language Models (LLMs) and Retrieval Augmented Generation (RAG) through Langchain, this app seamlessly translate natural 
language inquiries concerning financial statements into precise Snowflake queries. This innovative approach empowers users to explore complex financial data with simple questions, streamlining access to insightful analytics.

The information is stored for companies Apple, American Express, Bank Of America, Berkshire Hathaway, Johnson & Johnson, Coca-Cola Co, Mastercard Inc, Moody's Corp, 
Procter & Gamble Co and Verizon Communications Inc.
**Example questions to ask:**

- What was Proctor and Gamble's net income from 2010 through 2020?
- What was Apple's debt to equity ratio for the last 5 years?
- Rank the companies in descending order based on their net income in 2022. Include the ticker and net income value
- What has been the average for total assets and total liabilities for each company over the last 3 years? List the top 3
""")

# Chat window visibility depends on OpenAI key entry
if is_valid_key(open_ai_key):

    str_input = st.text_input(
        'What would you like to answer? (e.g. What was the revenue and net income for Apple for the last 5 years?)')

    if len(str_input) > 1:
        with st.spinner('Looking up your question in Snowflake now...'):
            # Here you would include your logic to handle the question, querying Snowflake
            # Since the fs_chain and sf_query functions are not defined, ensure to implement or adapt them according to your actual logic
            try:
                output = fs_chain(str_input)
                print("output", output)
                # st.write(output)
                try:
                    # if the output doesn't work we will try one additional attempt to fix it
                    query_result = sf_query(output['result'])
                    print("query result", query_result)
                    if len(query_result) > 1:
                        st.write(query_result)
                        st.write(output)
                except:
                    st.write("The first attempt didn't pull what you were needing. Trying again...")
                    output = fs_chain(
                        f'You need to fix the code but ONLY produce SQL code output. If the question is complex, consider using one or more CTE. Examine the DDL statements and answer this question: {output}')
                    st.write(sf_query(output['result']))
                    st.write(output)
            except:
                st.write(
                    "Please try to improve your question. Note this tab is for financial statement questions. Use Tab 3 to ask from shareholder letters. Also, only a handful of companies are available, which you can see on the side bar.")
                st.write(f"Final errored query used:")
                st.write(output)
else:
    st.warning("Please enter a valid OpenAI API key to enable the chat window.")
