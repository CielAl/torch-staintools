"""borrow Staintool's test cases
"""
import unittest
import numpy as np
from tests.util import fix_seed, dummy_from_numpy, psnr
from torch_staintools.functional.conversion.lab import rgb_to_lab
from torch_staintools.functional.stain_extraction.macenko import MacenkoExtractor
from torch_staintools.functional.stain_extraction.vahadane import VahadaneExtractor
from torch_staintools.functional.optimization.dict_learning import get_concentrations
from torch_staintools.functional.tissue_mask import get_tissue_mask, TissueMaskException
from torch_staintools.functional.utility.implementation import transpose_trailing, img_from_concentration
from functools import partial
from torchvision.transforms.functional import convert_image_dtype
from skimage.util import img_as_float32
import torch
import cv2
import os

fix_seed(314159)

# get_stain_mat = partial(MacenkoExtractor(), luminosity_threshold=None, num_stains=2, regularizer=0.1)


class TestFunctional(unittest.TestCase):
    device: torch.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device(0)

    rand_img = torch.randint(0, 255, (1, 3, 256, 256))

    THRESH_PSNR = 20

    @staticmethod
    def get_dummy_path():
        DUMMY_DIR = 'dummy_data'
        DUMMY_FILE = 'sample.png'
        # DUMMY_DATA_PATH = os.path.join('tests/images', DUMMY_DIR, DUMMY_FILE)
        return os.path.join(os.path.dirname(__file__), DUMMY_DIR, DUMMY_FILE)

    DUMMY_IMG_UBYTE = cv2.cvtColor(cv2.imread(get_dummy_path()), cv2.COLOR_BGR2RGB)
    DUMMY_IMG_TENSOR = dummy_from_numpy(DUMMY_IMG_UBYTE)

    @staticmethod
    def new_dummy_img_tensor_ubyte():
        return TestFunctional.DUMMY_IMG_TENSOR.clone()

    @staticmethod
    def stain_extract(dummy_tensor, get_stain_mat, luminosity_threshold, num_stains, algorithm, regularizer):

        # lab_tensor = rgb_to_lab(convert_image_dtype(dummy_tensor))

        stain_matrix = get_stain_mat(image=dummy_tensor, luminosity_threshold=luminosity_threshold,
                                     num_stains=num_stains, regularizer=regularizer)

        concentration = get_concentrations(dummy_tensor, stain_matrix, algorithm=algorithm,
                                           regularizer=regularizer)
        c_transposed_src = transpose_trailing(concentration)
        reconstructed = img_from_concentration(c_transposed_src, stain_matrix, dummy_tensor.shape, (0, 1))
        return stain_matrix, concentration, c_transposed_src, reconstructed

    @staticmethod
    def extract_eval_helper(tester, get_stain_mat, luminosity_threshold,
                            num_stains, regularizer, dict_algorithm):
        device = TestFunctional.device
        dummy_tensor_ubyte = TestFunctional.new_dummy_img_tensor_ubyte().to(device)
        # get_stain_mat = MacenkoExtractor()
        result_tuple = TestFunctional.stain_extract(dummy_tensor_ubyte, get_stain_mat,
                                                    luminosity_threshold=luminosity_threshold,
                                                    num_stains=num_stains,
                                                    algorithm=dict_algorithm, regularizer=regularizer)

        stain_matrix, concentration, c_transposed_src, reconstructed = result_tuple
        dummy_scaled = convert_image_dtype(dummy_tensor_ubyte, torch.float32)
        psnr_out = psnr(dummy_scaled, reconstructed).item()
        tester.assertTrue(psnr_out > TestFunctional.THRESH_PSNR)
        # size
        batch_size, channel_size, height, width = dummy_tensor_ubyte.shape
        tester.assertTrue(stain_matrix.shape == (batch_size, num_stains, channel_size))

        # transpose
        tester.assertTrue((c_transposed_src.permute(0, 2, 1) == concentration).all())

        # manual tissue mask
        mask = get_tissue_mask(dummy_scaled, luminosity_threshold=luminosity_threshold)
        tissue_count = mask.sum()
        tester.assertTrue(concentration.shape[-1] == tissue_count)

    def eval_wrapper(self, extractor):

        # all pixel
        algorithms = ['ista', 'cd']
        for alg in algorithms:
            TestFunctional.extract_eval_helper(self, extractor, luminosity_threshold=None,
                                               num_stains=2, regularizer=0.1, dict_algorithm=alg)

    def test_stains(self):
        macenko = MacenkoExtractor()
        vahadane = VahadaneExtractor()
        # not support num_stains other than 2
        with self.assertRaises(NotImplementedError):
            TestFunctional.extract_eval_helper(self, macenko, luminosity_threshold=None,
                                               num_stains=3, regularizer=0.1, dict_algorithm='ista')

        self.eval_wrapper(macenko)
        self.eval_wrapper(vahadane)

    def test_tissue_mask(self):
        device = TestFunctional.device
        dummy_scaled = convert_image_dtype(TestFunctional.new_dummy_img_tensor_ubyte(), torch.float32).to(device)
        mask = get_tissue_mask(dummy_scaled, luminosity_threshold=None)
        tissue_sum = mask.sum()
        self.assertTrue(tissue_sum == dummy_scaled.shape[-1] * dummy_scaled.shape[-2])

        mask = get_tissue_mask(dummy_scaled, luminosity_threshold=0.8, throw_error=True)
        self.assertTrue(mask.sum() > 0)

        with self.assertRaises(TissueMaskException):
            get_tissue_mask(torch.zeros_like(dummy_scaled), luminosity_threshold=0.8, throw_error=True)
