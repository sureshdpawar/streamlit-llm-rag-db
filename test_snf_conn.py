import streamlit as st
import snowflake.connector
from snowflake.connector import DictCursor

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

# Example query using the connection
def run_query(query):
    with conn.cursor(DictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()

# Example usage
query_result = run_query("SELECT CURRENT_VERSION()")
st.write(query_result)
