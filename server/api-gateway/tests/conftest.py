"""Test configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Mock authentication headers."""
    return {"Authorization": "Bearer test-token"}
