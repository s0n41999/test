import pandas as pd
import boto3
from io import StringIO
import streamlit as st
from datetime import datetime

# Replace with your bucket name and region
BUCKET_NAME = 'bucketstreamlit'
AWS_REGION = 'eu-north-1'

# Sample prediction data (replace with your actual prediction logic)
predictions = {'Date': ['2024-10-15', '2024-10-16'], 'Predicted Price': [150.25, 152.75]}
df = pd.DataFrame(predictions)

# Generate a unique filename using the current timestamp
filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Convert the DataFrame to a CSV format
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

# Upload the CSV to S3 with a unique filename
s3 = boto3.client('s3', region_name=AWS_REGION)
s3.put_object(Bucket=BUCKET_NAME, Key=filename, Body=csv_buffer.getvalue())

# Success message
st.success(f'Predictions have been saved to S3 as {filename}!')
