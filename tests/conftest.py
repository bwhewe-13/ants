
import pytest

@pytest.fixture(params=['00','25','50','100'])
def enrich(request):
    return request.param

@pytest.fixture(params=[80, 60, 43, 21, 10])
def g087r(request):
    return request.param

@pytest.fixture(params=[300, 240, 180, 150, 120, 90, 60, 30])
def g361r(request):
    return request.param

@pytest.fixture(params=[600, 500, 400, 300, 200, 100])
def g618r(request):
    return request.param
