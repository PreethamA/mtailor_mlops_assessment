import base64

from predictor import predict

def test_predict():
    with open("n01440764_tench.jpeg", "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    pred = predict(img_b64)
    print(f"Predicted class for tench image: {pred}")

if __name__ == "__main__":
    test_predict()

