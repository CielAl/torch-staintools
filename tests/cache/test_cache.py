import unittest
from torch_staintools.cache.tensor_cache import TensorCache
import torch
import os
from tests.util import fix_seed


fix_seed(314159)


class TestTensorCache(unittest.TestCase):
    device: torch.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(0)

    io_dirname = 'io_temp'
    io_dir = os.path.join(os.path.dirname(__file__), io_dirname)

    def test_instantiation(self):
        size_limit = -1
        tensor_cache = TensorCache(size_limit=size_limit,  device=None)
        self.assertTrue(tensor_cache.device == torch.device('cpu'))
        self.assertTrue(len(tensor_cache) == 0)
        self.assertTrue(isinstance(tensor_cache.data_cache, dict))
        # noinspection PyUnresolvedReferences
        self.assertTrue(tensor_cache._Cache__size_limit == size_limit)

    @staticmethod
    def new_dummy_tensors(num_elements):
        return torch.randn(num_elements, 3, 256, 256)

    @staticmethod
    def new_dummy_keys(start, num_elements):
        return [str(idx) for idx in range(start, start + num_elements)]

    @staticmethod
    def new_test_cache(size_limit):
        return TensorCache(size_limit=size_limit,  device=None)

    def test_batch(self):
        size_limit = 10
        num_elements = 10
        tensor_cache = TestTensorCache.new_test_cache(size_limit)
        self.assertTrue(len(tensor_cache) == 0)

        dummy_tensors = TestTensorCache.new_dummy_tensors(num_elements)
        dummy_keys = TestTensorCache.new_dummy_keys(0, num_elements)
        tensor_cache.write_batch(dummy_keys, dummy_tensors)

        cache_out = tensor_cache.get_batch_hit(dummy_keys)
        cache_out_stack = cache_out
        self.assertTrue((cache_out_stack == dummy_tensors).all())

        self.assertTrue(len(tensor_cache) == num_elements)
        for key in dummy_keys:
            self.assertTrue(tensor_cache.is_cached(key))
            self.assertTrue(key in tensor_cache)

        dummy_keys_exceed = TestTensorCache.new_dummy_keys(num_elements, num_elements)
        tensor_cache.write_batch(dummy_keys_exceed, dummy_tensors)

        self.assertTrue(len(tensor_cache) == num_elements)

        for key in dummy_keys_exceed:
            self.assertFalse(tensor_cache.is_cached(key))
            self.assertFalse(key in tensor_cache)

        with self.assertRaises(AssertionError):
            tensor_cache.get_batch_hit(dummy_keys_exceed)

    def test_single(self):
        size_limit = 1
        tensor_cache = TestTensorCache.new_test_cache(size_limit)
        dummy_tensor_single = TestTensorCache.new_dummy_tensors(1)[0]
        key_single = TestTensorCache.new_dummy_keys(0, 1)[0]

        tensor_cache.write_to_cache(key_single, dummy_tensor_single)
        self.assertTrue(tensor_cache.is_cached(key_single))
        self.assertTrue(key_single in tensor_cache)

        out = tensor_cache.get(key_single, func=None)
        # noinspection PyUnresolvedReferences
        self.assertTrue((out == dummy_tensor_single).all())

        key_exceed = TestTensorCache.new_dummy_keys(1, 1)[0]
        tensor_cache.write_to_cache(key_exceed, dummy_tensor_single)
        self.assertFalse(tensor_cache.is_cached(key_exceed))
        self.assertFalse(key_exceed in tensor_cache)

        with self.assertRaises(AssertionError):
            tensor_cache.get(key_exceed, func=None)

    def test_key_type(self):
        size_limit = -1
        tensor_cache = TestTensorCache.new_test_cache(size_limit)
        with self.assertRaises(TypeError):
            unhashable_key = [1, 2, 3]
            # noinspection PyTypeChecker
            tensor_cache.write_to_cache(key=unhashable_key, value=torch.ones(1))

    def test_lazy_eval(self):
        size_limit = 100
        tensor_cache = TestTensorCache.new_test_cache(size_limit)
        key_single = TestTensorCache.new_dummy_keys(0, 1)[0]

        dummy_size = (1, 3, 256, 256)
        dummy_tensor_single_determinated = torch.ones(*dummy_size)
        # get with lazy eval
        lazy_out = tensor_cache.get(key_single, torch.ones, *dummy_size)
        self.assertTrue((lazy_out == dummy_tensor_single_determinated).all())

        self.assertTrue(tensor_cache.is_cached(key_single))
        self.assertTrue(key_single in tensor_cache)

        out_no_eval = tensor_cache.get(key_single, func=None)
        self.assertTrue(out_no_eval is lazy_out)

    @staticmethod
    def examine_tensor_helper(tester, tensor_cache, device):
        for v in tensor_cache.data_cache.values():
            tester.assertTrue(isinstance(v, torch.Tensor))
            tester.assertTrue(v.device == device)

    def test_value(self):
        size_limit = -1
        tensor_cache = TestTensorCache.new_test_cache(size_limit)
        num_elements = 5
        dummy_tensors = TestTensorCache.new_dummy_tensors(num_elements)
        dummy_keys = TestTensorCache.new_dummy_keys(0, num_elements)
        tensor_cache.write_batch(dummy_keys, dummy_tensors)
        TestTensorCache.examine_tensor_helper(self, tensor_cache, torch.device('cpu'))

        tensor_cache.to(TestTensorCache.device)
        TestTensorCache.examine_tensor_helper(self, tensor_cache, TestTensorCache.device)

    def test_io(self):
        os.makedirs(TestTensorCache.io_dir, exist_ok=True)
        dump_size = 25
        tensor_cache = TestTensorCache.new_test_cache(-1)
        dump_tensor = TestTensorCache.new_dummy_tensors(dump_size)
        dump_key = '1'
        tensor_cache.write_to_cache(dump_key, dump_tensor)

        file = os.path.join(TestTensorCache.io_dir, 'dummy_cache.pt')
        tensor_cache.dump(file, force_overwrite=True)
        self.assertTrue(os.path.exists(file))

        another_cache = TestTensorCache.new_test_cache(-1)
        different_dump_tensor = TestTensorCache.new_dummy_tensors(dump_size * 2)
        # definitely different
        another_cache.write_to_cache(dump_key, different_dump_tensor)
        another_cache.load(file)
        for k, v in tensor_cache.data_cache.items():
            self.assertTrue((v.shape == another_cache.data_cache[k].shape) and
                            (v == another_cache.data_cache[k]).all())
        # check whether the load cause the update
        self.assertFalse(another_cache.get(dump_key, func=None).shape == different_dump_tensor.shape)
