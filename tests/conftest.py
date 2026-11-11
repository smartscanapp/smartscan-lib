
import pytest


def pytest_addoption(parser):
    parser.addoption("--images", nargs="+", default=[], help="Image paths")
    parser.addoption("--items", type=int, default=10_000, help="Number of items to benchmark")
    parser.addoption("--dim", type=int, default=384, help="Embedding dimensions")


@pytest.fixture
def images(request):
    return request.config.getoption("images")


@pytest.fixture
def n_items(request):
    return request.config.getoption("items")
