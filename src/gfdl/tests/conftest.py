import os
import pytest
from sklearn import set_config

def pytest_configure(config):
    os.environ["SCIPY_ARRAY_API"] = "1"

@pytest.fixture(autouse=True)
def enable_array_api():
    # Turn it on before the test runs
    set_config(array_api_dispatch=True)
    yield
    # Reset it back to default after the test finishes
    set_config(array_api_dispatch=False)
