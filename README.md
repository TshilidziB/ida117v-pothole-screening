# South African Pothole Image Screening

This Streamlit application demonstrates the selected MLP model from the IDA117V group project. It accepts one JPG, JPEG or PNG road image and predicts **pothole** or **no pothole**.

The model uses HOG, RGB colour-histogram and Sobel edge features extracted from a resized 128 x 128 RGB image. The saved bundle contains the fitted training-only scaler, the selected MLP and the classification threshold.

This is a student demonstration and screening aid. It does not estimate pothole severity, depth, location, repair cost or repair priority. Important decisions should be reviewed by a person.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
