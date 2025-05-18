# tests/test_api.py

import os
import requests
import pytest

# Base URL for the API under test
BASE_URL = os.getenv("API_URL", "http://localhost:8000")
# Path to a sample PDF for testing
TEST_PDF_PATH = os.path.join(os.path.dirname(__file__), "test.pdf")


def test_full_workflow():
    # 1. Create vector store
    resp = requests.post(
        f"{BASE_URL}/vector_stores", json={
            "name": "pytest_store",
            "dimension": 4096,
            "distance": "Cosine"
        }
    )
    assert resp.status_code == 201, f"Create store failed: {resp.text}"
    data = resp.json()
    store_id = data.get("id")
    assert store_id, "Store ID not returned"
    assert data.get("name") == "pytest_store"

    # 2. List vector stores
    resp = requests.get(f"{BASE_URL}/vector_stores")
    assert resp.status_code == 200, f"List stores failed: {resp.text}"
    stores = resp.json()
    assert any(s.get("id") == store_id for s in stores)

    # 3. Retrieve vector store
    resp = requests.get(f"{BASE_URL}/vector_stores/{store_id}")
    assert resp.status_code == 200, f"Retrieve store failed: {resp.text}"
    store = resp.json()
    assert store.get("id") == store_id

    # 4. Update vector store name
    resp = requests.patch(
        f"{BASE_URL}/vector_stores/{store_id}", json={"name": "updated_pytest"}
    )
    assert resp.status_code == 200, f"Update store failed: {resp.text}"
    updated = resp.json()
    assert updated.get("name") == "updated_pytest"

    # 5. Upload a sample PDF file
    assert os.path.exists(TEST_PDF_PATH), "test.pdf not found in tests directory"
    with open(TEST_PDF_PATH, "rb") as pdf_file:
        files = {"file": ("test.pdf", pdf_file, "application/pdf")}
        resp = requests.post(
            f"{BASE_URL}/vector_stores/{store_id}/files", files=files
        )
    assert resp.status_code == 201, f"File upload failed: {resp.text}"
    file_data = resp.json()
    file_id = file_data.get("file_id")
    assert file_id, "File ID not returned"
    # Expect at least one page extracted from PDF
    assert isinstance(file_data.get("pages"), int) and file_data.get("pages") >= 1

    # 6. List ingested files
    resp = requests.get(f"{BASE_URL}/vector_stores/{store_id}/files")
    assert resp.status_code == 200, f"List files failed: {resp.text}"
    files_list = resp.json()
    assert file_id in files_list

    # 7. Retrieve file metadata
    resp = requests.get(f"{BASE_URL}/vector_stores/{store_id}/files/{file_id}")
    assert resp.status_code == 200, f"Get file metadata failed: {resp.text}"
    meta = resp.json()
    assert meta.get("filename") == "test.pdf"

    # 8. Query vectors
    resp = requests.post(
        f"{BASE_URL}/vector_stores/{store_id}/query", json={
            "query": "test",
            "top_k": 1
        }
    )
    assert resp.status_code == 200, f"Query failed: {resp.text}"
    results = resp.json()
    assert isinstance(results, list), "Query did not return list"

    # 9. Delete ingested file
    resp = requests.delete(
        f"{BASE_URL}/vector_stores/{store_id}/files/{file_id}"
    )
    assert resp.status_code == 200, f"Delete file failed: {resp.text}"
    assert resp.json().get("status") == "deleted"

    # 10. Delete vector store
    resp = requests.delete(f"{BASE_URL}/vector_stores/{store_id}")
    assert resp.status_code == 200, f"Delete store failed: {resp.text}"
    assert resp.json().get("status") == "deleted"


if __name__ == "__main__":
    pytest.main([__file__])
