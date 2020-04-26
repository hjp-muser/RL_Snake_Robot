import unittest
import random
import numpy as np
import torch


class TestRandomUtils(unittest.TestCase):
    def test_set_global_seeds(self):
        set_global_seeds(1)
        a = random.random()
        set_global_seeds(1)
        b = random.random()
        self.assertEqual(a, b)
        set_global_seeds(1)
        a = np.random.random()
        set_global_seeds(1)
        b = np.random.random()
        self.assertEqual(a, b)
        set_global_seeds(1)
        a = torch.randint(1, 10, (1,))
        set_global_seeds(1)
        b = torch.randint(1, 10, (1,))
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()

