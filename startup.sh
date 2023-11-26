#!/bin/bash
nginx -t &&
service nginx start &&
python app/chunk_1.py 
streamlit run app/app.py