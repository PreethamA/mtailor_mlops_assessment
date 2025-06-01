import argparse
import base64
import requests
import sys

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def send_request(api_url, api_key, image_path):
    img_b64 = encode_image(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "image_base64": img_b64
    }
    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        print("Prediction result:", response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")

def run_predefined_tests(api_url, api_key):
    test_images = [
        ("n01440764_tench.jpeg", 0),
        ("n01667114_mud_turtle.JPEG", 35)
    ]
    for img, expected_class in test_images:
        print(f"\nTesting {img} (expect class {expected_class})")
        send_request(api_url, api_key, img)

def main():
    parser = argparse.ArgumentParser(description="Test deployed Cerebrium model API.")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--url", type=str, required=True, help="API URL of deployed model")
    parser.add_argument("--api-key", type=str, required=True, help="API Key for authentication")
    parser.add_argument("--run-tests", action="store_true", help="Run preset tests")

    args = parser.parse_args()

    if args.run_tests:
        run_predefined_tests(args.url, args.api_key)
    elif args.image:
        send_request(args.url, args.api_key, args.image)
    else:
        print("Please specify either --image or --run-tests")
        sys.exit(1)

if __name__ == "__main__":
    main()

