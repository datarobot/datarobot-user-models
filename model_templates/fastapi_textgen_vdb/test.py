import requests
import json
import io

# Server URL
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_predict_endpoint():
    """Test the /predict endpoint with text data"""
    print("\n=== Testing /predict Endpoint with Text Data ===")
    
    # Test with plain text
    query = "clustering algorithm"
    headers = {"content-type": "text/plain"}
    response = requests.post(
        f"{BASE_URL}/predict",
        data=query.encode("utf-8"),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_predict_with_csv():
    """Test the /predict endpoint with CSV file upload"""
    print("\n=== Testing /predict Endpoint with CSV Upload ===")
    
    # Create a simple CSV with a query
    csv_content = "query\nrandom forest"
    files = {"file": ("query.csv", io.StringIO(csv_content), "text/csv")}
    
    response = requests.post(
        f"{BASE_URL}/predict",
        files=files
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_predict_unstructured():
    """Test the /predictUnstructured endpoint"""
    print("\n=== Testing /predictUnstructured Endpoint ===")
    
    # Test with plain text
    query = "support vector machine"
    headers = {"content-type": "text/plain"}
    response = requests.post(
        f"{BASE_URL}/predictUnstructured",
        data=query.encode("utf-8"),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")
    
    # Test with JSON input
    print("\n--- Testing /predictUnstructured with JSON ---")
    data = ["decision tree"]
    headers = {"content-type": "application/json"}
    response = requests.post(
        f"{BASE_URL}/predictUnstructured",
        data=json.dumps(data),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_chat_completions():
    """Test the /chat/completions endpoint"""
    print("\n=== Testing /chat/completions Endpoint ===")
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about linear regression"}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_tools_call():
    """Test the /tools/call endpoint"""
    print("\n=== Testing /tools/call Endpoint ===")
    
    data = {
        "id": 123,
        "method": "tools/call",
        "params": {
            "name": "scikit_search",
            "arguments": {
                "query": "gradient boosting"
            }
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/tools/call",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

if __name__ == "__main__":
    print("Starting tests for Scikit-learn FAISS Retrieval Server...")
    
    # Run all tests
    test_root_endpoint()
    test_predict_endpoint()
    # test_predict_with_csv()
    # test_predict_unstructured()
    # test_chat_completions()
    # test_tools_call()
    
    print("\nAll tests completed!")