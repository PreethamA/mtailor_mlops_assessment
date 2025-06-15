import requests
import json
import argparse
import os
from PIL import Image
import io
import time

# Placeholder for Cerebrium specific URLs or headers if needed
# CEREBRIUM_PREDICT_ENDPOINT_SUFFIX = "/predict" # Example

def call_deployed_model(api_link: str, api_key: str, image_path: str = None, test_mode: bool = False):
    """
    Makes a POST request to the deployed model's prediction endpoint.

    Args:
        api_link (str): The base URL of the deployed model's API endpoint.
        api_key (str): The API key for authentication.
        image_path (str, optional): Path to the image file to send.
        test_mode (bool): If True, indicates this is part of a larger test suite.
                           Adjusts output for cleaner integration.

    Returns:
        dict: The JSON response from the model API, or an error message.
    """
    # Assuming Cerebrium uses an Authorization header
    headers = {
        'Authorization': f'Bearer {api_key}',
        'accept': 'application/json' # Standard header
    }

    if not image_path or not os.path.exists(image_path):
        error_msg = f"Error: Image file not found at {image_path}" if image_path else "No image path provided."
        print(error_msg)
        return {"error": error_msg}, 0

    try:
        with open(image_path, 'rb') as f:
            # For file uploads, requests uses 'files' parameter
            # The field name 'file' must match what 'app.py' expects (request.files['file'])
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')} # Or 'image/png' etc.
            if not test_mode:
                print(f"Sending image file: {image_path} to {api_link}")
            start_time = time.time()
            response = requests.post(api_link, headers=headers, files=files, timeout=30) # Add a timeout
            end_time = time.time()
            inference_time = end_time - start_time
            if not test_mode:
                print(f"Inference time: {inference_time:.4f} seconds")

        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        response_json = response.json()
        if not test_mode:
            print("\n--- Raw API Response ---")
            print(json.dumps(response_json, indent=4))
            print("------------------------\n")
        return response_json, inference_time

    except requests.exceptions.HTTPError as errh:
        error_details = {"error": f"HTTP Error: {errh}", "status_code": errh.response.status_code, "response_text": errh.response.text}
        if not test_mode: print(f"Http Error: {errh}\nResponse: {errh.response.text}")
        return error_details, 0
    except requests.exceptions.ConnectionError as errc:
        error_details = {"error": f"Error Connecting: {errc}"}
        if not test_mode: print(f"Error Connecting: {errc}")
        return error_details, 0
    except requests.exceptions.Timeout as errt:
        error_details = {"error": f"Timeout Error: {errt}"}
        if not test_mode: print(f"Timeout Error: {errt}")
        return error_details, 0
    except requests.exceptions.RequestException as err:
        error_details = {"error": f"An unexpected request error occurred: {err}"}
        if not test_mode: print(f"OOps: Something Else {err}")
        return error_details, 0
    except json.JSONDecodeError:
        error_details = {"error": "Invalid JSON response", "response_text": response.text}
        if not test_mode: print(f"Error decoding JSON response: {response.text}")
        return error_details, 0
    except Exception as e:
        error_details = {"error": f"An unexpected error occurred during API call: {e}"}
        if not test_mode: print(f"General error: {e}")
        return error_details, 0


def run_preset_custom_tests(api_link: str, api_key: str):
    """
    Runs a series of pre-defined tests against the deployed model.
    """
    print("\n--- Running Preset Custom Tests on Deployed Model ---")

    # Sample images and their expected class IDs
    test_cases = [
        {"path": "data/n01440764_tench.jpeg", "expected_id": 0, "name": "Tench Fish"},
        {"path": "data/n01667114_mud_turtle.jpeg", "expected_id": 35, "name": "Mud Turtle"}
    ]

    all_tests_passed = True
    total_inference_time = 0
    num_successful_inferences = 0

    for i, test_case in enumerate(test_cases):
        print(f"\nTest {i+1}: Testing with {test_case['name']} ({test_case['path']})")
        if not os.path.exists(test_case['path']):
            print(f"  Warning: Test image not found at {test_case['path']}. Skipping this test case.")
            all_tests_passed = False
            continue

        response, inference_time = call_deployed_model(api_link, api_key, test_case['path'], test_mode=True)

        if "error" in response:
            print(f"  Test FAILED for {test_case['name']}: {response['error']}")
            all_tests_passed = False
        else:
            total_inference_time += inference_time
            num_successful_inferences += 1
            predicted_id = response.get('predicted_class_id')
            probabilities = response.get('probabilities')

            print(f"  Predicted Class ID: {predicted_id}, Expected ID: {test_case['expected_id']}")
            print(f"  Inference Time: {inference_time:.4f} seconds")

            # Check if prediction is correct
            if predicted_id == test_case['expected_id']:
                print(f"  Prediction for {test_case['name']} PASSED.")
            else:
                print(f"  Prediction for {test_case['name']} FAILED: Incorrect class ID.")
                all_tests_passed = False

            # Check for reasonable probability output (e.g., sum of probabilities close to 1)
            if probabilities and abs(sum(probabilities) - 1.0) < 1e-4:
                print(f"  Probability sum check PASSED.")
            elif probabilities:
                print(f"  Probability sum check FAILED: Sum is {sum(probabilities):.4f}.")
                all_tests_passed = False
            else:
                print(f"  Probability output missing or invalid.")
                all_tests_passed = False


    # Additional tests for platform monitoring / health
    print("\n--- Platform Monitoring Tests ---")
    health_link = api_link.rsplit('/', 1)[0] + "/health" # Assumes health endpoint at base path
    print(f"Checking health endpoint: {health_link}")
    try:
        health_response = requests.get(health_link, headers={'Authorization': f'Bearer {api_key}'}, timeout=10)
        health_response.raise_for_status()
        health_data = health_response.json()
        print(f"  Health Check Status: {health_data.get('status')}")
        print(f"  Health Check Message: {health_data.get('message')}")
        if health_data.get('status') == 'healthy':
            print("  Health check PASSED.")
        else:
            print("  Health check FAILED: Status not 'healthy'.")
            all_tests_passed = False
    except Exception as e:
        print(f"  Health check FAILED: Could not reach health endpoint or parse response: {e}")
        all_tests_passed = False

    print("\n--- End of Preset Custom Tests ---")
    if all_tests_passed:
        print("ALL DEPLOYMENT TESTS PASSED!")
        if num_successful_inferences > 0:
            print(f"Average Inference Time: {(total_inference_time / num_successful_inferences):.4f} seconds (for {num_successful_inferences} successful inferences)")
        return True
    else:
        print("SOME DEPLOYMENT TESTS FAILED. Please review the logs.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a deployed machine learning model on Cerebrium.")
    parser.add_argument("--api_link", type=str, required=True,
                        help="The URL of the deployed model's API endpoint (e.g., https://your-cerebrium-model.run/predict).")
    parser.add_argument("--api_key", type=str, required=True,
                        help="The API key for authentication with the deployed model.")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to an image file to send for a single prediction test.")
    parser.add_argument("--run_preset_tests", action="store_true",
                        help="Flag to run a series of preset custom tests on the deployed model.")

    args = parser.parse_args()

    # Ensure data directory and sample images exist for preset tests
    if args.run_preset_tests:
        if not os.path.exists("data"):
            print("Error: 'data' directory not found. Please create it and place sample images.")
            exit(1)
        if not os.path.exists("data/n01440764_tench.jpeg") or not os.path.exists("data/n01667114_mud_turtle.jpeg"):
            print("Error: Missing sample images in 'data' directory. Please ensure 'n01440764_tench.jpeg' and 'n01667114_mud_turtle.jpeg' are present.")
            exit(1)

    if args.run_preset_tests:
        run_preset_custom_tests(args.api_link, args.api_key)
    elif args.image_path:
        print("\n--- Starting Single Prediction Test ---")
        response, inference_time = call_deployed_model(args.api_link, args.api_key, args.image_path)

        if "error" in response:
            print(f"\nSingle Prediction Test FAILED: {response['error']}")
            if "status_code" in response:
                print(f"HTTP Status Code: {response['status_code']}")
            if "response_text" in response:
                print(f"Response Text: {response['response_text']}")
        else:
            predicted_class_id = response.get('predicted_class_id')
            print(f"\n--- Single Prediction Result ---")
            print(f"Image Path: {args.image_path}")
            print(f"Predicted Class ID: {predicted_class_id}")
            print(f"Inference Time: {inference_time:.4f} seconds")
            print("--------------------------------\n")
            print("Single Prediction Test PASSED (check output for correctness).")
    else:
        print("Please specify either --image_path for a single test or --run_preset_tests for multiple tests.")
        parser.print_help()
        exit(1)