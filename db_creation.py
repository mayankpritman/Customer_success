import sqlite3
import pandas as pd
conn = sqlite3.connect("customer_success.db")
pd.read_csv("customers.csv").to_sql("customers", conn, if_exists="replace", index=False)
pd.read_csv("tickets.csv").to_sql("tickets", conn, if_exists="replace", index=False)
pd.read_csv("interactions.csv").to_sql("interactions", conn, if_exists="replace", index=False)
pd.read_csv("nps.csv").to_sql("nps", conn, if_exists="replace", index=False)
conn.close()