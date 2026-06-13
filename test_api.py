#!/usr/bin/env python
"""
Test script for CLV API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    print(f"  ✅ Health: {data['status']}")
    print(f"     Models loaded: {data['models_loaded']}")
    return data


def test_predict_clv():
    """Test CLV prediction."""
    print("\nTesting /predict/clv...")
    
    payload = {
        "customer_id": "12345",
        "frequency": 5,
        "recency": 30.0,
        "T": 180.0,
        "monetary_value": 150.0,
        "n_unique_products": 20,
        "country": "UK"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/clv",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    print(f"  ✅ Prediction for customer {data['customer_id']}")
    print(f"     Predicted CLV (90d): £{data['predicted_clv_90d']:.2f}")
    print(f"     Predicted CLV (12m): £{data['predicted_clv_12m']:.2f}")
    print(f"     Probability Alive: {data['prob_alive']:.2%}")
    print(f"     RFM Segment: {data['rfm_segment']}")
    print(f"     CLV Segment: {data['clv_segment']}")
    print(f"     Action: {data['recommended_action']}")
    
    return data


def test_segments():
    """Test segments endpoint."""
    print("\nTesting /segments...")
    
    response = requests.get(f"{BASE_URL}/segments")
    assert response.status_code == 200
    
    segments = response.json()
    print(f"  ✅ Found {len(segments)} segments")
    
    for seg in segments[:5]:
        print(f"     - {seg['segment']}: {seg['customer_count']} customers, "
              f"{seg['revenue_share_pct']}% revenue, priority: {seg['priority']}")
    
    return segments


def test_batch_predict():
    """Test batch prediction."""
    print("\nTesting /predict/batch...")
    
    customers = [
        {"customer_id": f"{i}", "frequency": i % 10, "recency": i * 10, 
         "T": 100, "monetary_value": 50 + i * 10, "n_unique_products": 5, "country": "UK"}
        for i in range(1, 6)
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=customers,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    predictions = response.json()
    print(f"  ✅ Batch prediction: {len(predictions)} customers")
    
    return predictions


def test_top_customers():
    """Test top customers endpoint."""
    print("\nTesting /top-customers...")
    
    response = requests.get(f"{BASE_URL}/top-customers?limit=10")
    assert response.status_code == 200
    
    customers = response.json()
    print(f"  ✅ Top 10 customers by CLV:")
    for cust in customers[:5]:
        print(f"     - Customer {cust['customer_id']}: £{cust['predicted_clv']:.2f}")
    
    return customers


def test_at_risk():
    """Test at-risk customers endpoint."""
    print("\nTesting /at-risk...")
    
    response = requests.get(f"{BASE_URL}/at-risk?limit=10")
    assert response.status_code == 200
    
    customers = response.json()
    print(f"  ✅ At-risk customers: {len(customers)}")
    for cust in customers[:3]:
        print(f"     - Customer {cust['customer_id']}: CLV £{cust['predicted_clv']:.2f}, "
              f"prob_alive {cust['prob_alive']:.2%}")
    
    return customers


def main():
    print("=" * 60)
    print("🧪 CLV API Test Suite")
    print("=" * 60)
    
    # Wait for API to be ready
    print("\nWaiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("❌ API not responding. Make sure it's running on port 8000")
        return
    
    # Run tests
    try:
        test_health()
        test_predict_clv()
        test_segments()
        test_batch_predict()
        test_top_customers()
        test_at_risk()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()