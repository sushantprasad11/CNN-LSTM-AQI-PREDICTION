import joblib

p = r"D:\AQI-Prediction-CNN-LSTM\artifacts\scaler.joblib"

obj = joblib.load(p)

print("Dictionary keys:", list(obj.keys()))

for k, v in obj.items():
    print("\nKey:", k)
    print("Value type:", type(v))
    
    if isinstance(v, dict):
        print("Nested keys:", list(v.keys()))
    
    elif hasattr(v, "n_features_in_"):
        print("Features:", v.n_features_in_)
    
    elif isinstance(v, (list, tuple)):
        print("Length:", len(v))
        print("First 10 items:", v[:10])
    
    else:
        print("Value:", repr(v)[:500])
