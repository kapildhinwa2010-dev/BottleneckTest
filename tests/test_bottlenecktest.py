import pytest
import torch
from bottlenecktest import BottleneckTest, create_bottlenecktest

class TestBottleneckTest:
    def test_init(self):
        m = BottleneckTest(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = BottleneckTest(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_bottlenecktest()
        assert isinstance(m, BottleneckTest)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
