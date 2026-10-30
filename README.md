# Fashion Item Classifier - local demo

## Folder layout
```
fashion_app/
  app.py
  preprocess.py
  requirements.txt
  fashion_cnn.keras     <- downloaded from Colab (see notebook, Section 12)
  samples/              <- unzip samples.zip here (optional, for the "Pick a sample" mode)
```

## Run (Windows / Mac / Linux)
1. Install Python 3.10, 3.11 or 3.12.
2. Open a terminal in the `fashion_app` folder, then:
```
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
3. Your browser opens at http://localhost:8501

The first start is slow (TensorFlow loads). After that each prediction takes a fraction of a second.
