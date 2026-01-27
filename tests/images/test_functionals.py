"""borrow Staintool's test cases
"""
import unittest
from tests.util import fix_seed, dummy_from_numpy, psnr
from torch_staintools.functional.stain_extraction.macenko import MacenkoExtractor
from torch_staintools.functional.stain_extraction.vahadane import VahadaneExtractor
from torch_staintools.functional.optimization.concentration import get_concentrations
from torch_staintools.functional.tissue_mask import get_tissue_mask, TissueMaskException
from torch_staintools.functional.utility.implementation import transpose_trailing, img_from_concentration
from torchvision.transforms.functional import convert_image_dtype
from torch_staintools.normalizer.reinhard import ReinhardNormalizer
import torch
import os
import cv2

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
        algorithms = ['ista', 'cd', 'ls']
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

    @staticmethod
    def mean_std_compare_squeezed(x, mask):
        masked = x * mask
        mean_list = []
        std_list = []
        for c in range(masked.shape[1]):
            nonzero = masked[:, c: c + 1, :, :][mask != 0]
            mean_list.append(nonzero.mean())
            std_list.append(nonzero.std())
        return torch.stack(mean_list).squeeze(), torch.stack(std_list).squeeze()

    def test_reinhard(self):
        device = TestFunctional.device
        dummy_scaled = convert_image_dtype(TestFunctional.new_dummy_img_tensor_ubyte(), torch.float32).to(device)
        # not None mask
        mask = get_tissue_mask(dummy_scaled, luminosity_threshold=0.8)
        # 1 x 3 x 1 x 1

        means_input, stds_input = ReinhardNormalizer._mean_std_helper(dummy_scaled, mask=mask)

        manual_mean, manual_std = TestFunctional.mean_std_compare_squeezed(dummy_scaled, mask)
        self.assertTrue(torch.isclose(manual_mean, means_input.squeeze()).all())
        self.assertTrue(torch.isclose(manual_std, stds_input.squeeze()).all())

        # no mask
        rand_dummy = torch.randn(dummy_scaled.shape, device=dummy_scaled.device)
        rand_mean, rand_std = ReinhardNormalizer._mean_std_helper(rand_dummy, mask=None)

        rand_mean_truth = rand_dummy.mean(dim=(2, 3), keepdim=True)
        rand_std_truth = rand_dummy.std(dim=(2, 3), keepdim=True)

        self.assertTrue(torch.isclose(rand_mean, rand_mean_truth).all())
        self.assertTrue(torch.isclose(rand_std, rand_std_truth).all())

