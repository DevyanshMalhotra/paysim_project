# PaySim Fraud Detection

- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt
- place data/paysim.csv in the data/ folder
- python train.py
- python test_pipeline.py
- python api.py
- python simulator.py
- streamlit run app.py
- docker build -t paysim-fraud .
- docker run -p 8501:8501 -p 8000:8000 paysim-fraud
