# db_tool.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
DB_PATH = "customer_success.db"
def execute_query(query: str) -> pd.DataFrame:
   """
   Run an arbitrary SQL SELECT on customer_success.db and return a DataFrame.
   """
   with sqlite3.connect(DB_PATH) as conn:
       return pd.read_sql_query(query, conn)
def generate_chart(
   query: str,
   x_col: str,
   y_col: str,
   chart_type: str = "bar"
) -> dict:
   """
   Run the SQL, build a bar or line chart from x_col vs y_col,
   encode it as a base64 PNG, and return the payload.
   """
   df = execute_query(query)
   if df.empty or x_col not in df.columns or y_col not in df.columns:
       return {"type": "error", "message": "Invalid columns or empty result."}
   fig, ax = plt.subplots()
   if chart_type.lower() == "bar":
       ax.bar(df[x_col], df[y_col])
   elif chart_type.lower() == "line":
       ax.plot(df[x_col], df[y_col])
   else:
       return {"type": "error", "message": f"Unsupported chart type: {chart_type}"}
   ax.set_title(f"{chart_type.title()} of {y_col} vs {x_col}")
   buf = io.BytesIO()
   fig.tight_layout()
   plt.savefig(buf, format="png")
   buf.seek(0)
   img_base64 = base64.b64encode(buf.read()).decode()
   plt.close(fig)
   return {"type": "chart", "format": "png", "data": img_base64}