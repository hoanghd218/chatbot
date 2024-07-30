import pandas as pd
from sqlalchemy import create_engine

# Load the Excel file
df = pd.read_excel(r"C:\Users\Administrator\Desktop\a.xlsx", sheet_name='a',header=0)

date_columns = ['Date Ready for Collection', 'Date Returned','Date Collected or Sent']  # Replace with your actual column names

# Convert specified columns to datetime
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Create a SQLAlchemy engine
engine = create_engine("postgresql://do_invoice:wohhup2021@wh-idd-test-dev.c9lyw52w9grj.ap-southeast-1.rds.amazonaws.com:5432/ai_pgvector")

# Insert data into PostgreSQL
df.to_sql('a', engine, index=False, if_exists='replace')

# Optional: Verify the data is inserted
with engine.connect() as connection:
    result = connection.execute("SELECT * FROM a LIMIT 5")
    for row in result:
        print(row)

