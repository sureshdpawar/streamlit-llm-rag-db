import os
import glob
import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark.session import Session
import streamlit as st

# snowpark connection
CONNECTION_PARAMETERS = {
   "account": st.secrets['account'],
   "user": st.secrets['user'],
   "password": st.secrets['password'],
   "database": st.secrets['database'],
   "schema": st.secrets['schema'],
   "warehouse": st.secrets['warehouse'],
}

# Example of using a fully qualified name for a table
fully_qualified_table_name = f"{CONNECTION_PARAMETERS['database']}.{CONNECTION_PARAMETERS['schema']}.YOUR_TABLE_NAME"


# create session
session = Session.builder.configs(CONNECTION_PARAMETERS).create()

# create a list of the statements which should match the folder name
statements = ['INCOME_STATEMENT_ANNUAL','BALANCE_SHEET_ANNUAL','CASH_FLOW_STATEMENT_ANNUAL']

for statement in statements:
    path = f'./financials/{statement.lower()}/'
    files = glob.glob(os.path.join(path, "*.csv"))
    if files:  # Check if the list is not empty
        df = pd.concat((pd.read_csv(f) for f in files))
        print(statement)
        session.create_dataframe(df).write.mode('overwrite').save_as_table(statement)
    else:
        print(f"No CSV files found in {path}")


# automatically get the ddl from the created tables
# create empty string that will be populated
ddl_string = ''

# run through the statements and get ddl
for statement in statements:
    ddl_string += session.sql(f"select get_ddl('table', '{statement}')").collect()[0][0] + '\n\n'
    
ddl_file = open("ddls.sql", "w")
n = ddl_file.write(ddl_string)
ddl_file.close()
