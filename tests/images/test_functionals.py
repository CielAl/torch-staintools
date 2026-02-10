"""borrow Staintool's test cases
"""
import unittest
from typing import Optional, cast

from tests.util import fix_seed, dummy_from_numpy, psnr
from torch_staintools.constants import CONFIG
from torch_staintools.functional.conversion.od import rgb2od
from torch_staintools.functional.optimization.sparse_util import METHOD_FACTORIZE
from torch_staintools.functional.stain_extraction.extractor import StainExtraction
from torch_staintools.functional.stain_extraction.macenko import MacenkoAlg, DEFAULT_MACENKO_CONFIG
from torch_staintools.functional.stain_extraction.vahadane import VahadaneAlg, DEFAULT_VAHADANE_CONFIG
from torch_staintools.functional.concentration import ConcentrationSolver, ConcentCfg
from torch_staintools.functional.tissue_mask import get_tissue_mask, TissueMaskException
from torch_staintools.functional.utility.implementation import img_from_concentration
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
    POSITIVE_CONC_CFG = ConcentCfg()

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
        device = TestFunctional.device
        return TestFunctional.DUMMY_IMG_TENSOR.clone().to(device)

    @staticmethod
    def stain_extract(dummy_tensor: torch.Tensor, get_stain_mat: StainExtraction,
                      conc_solver: ConcentrationSolver,
                      luminosity_threshold: float,
                      mask: Optional[torch.Tensor],
                      rng: Optional[torch.Generator]):

        # lab_tensor = rgb_to_lab(convert_image_dtype(dummy_tensor))
        mask = get_tissue_mask(dummy_tensor, luminosity_threshold, mask)
        od_dummy = rgb2od(dummy_tensor)
        stain_matrix = get_stain_mat(od=od_dummy,
                                     mask=mask,
                                     )

        concentration = conc_solver(od_dummy, stain_matrix, rng=rng)
        reconstructed = img_from_concentration(concentration, stain_matrix, dummy_tensor.shape, (0, 1))
        return stain_matrix, concentration, reconstructed

    @staticmethod
    def extract_eval_helper(tester,
                            dummy_tensor_ubyte: torch.Tensor,
                            get_stain_mat: StainExtraction,
                            conc_solver: ConcentrationSolver,
                            luminosity_threshold: Optional[float],
                            mask: Optional[torch.Tensor],
                            rng: Optional[torch.Generator]):

        result_tuple = TestFunctional.stain_extract(dummy_tensor_ubyte, get_stain_mat,
                                                    conc_solver=conc_solver,
                                                    luminosity_threshold=luminosity_threshold,
                                                    mask=mask,
                                                    rng=rng)

        stain_matrix, concentration, reconstructed = result_tuple
        od_real = rgb2od(dummy_tensor_ubyte).flatten(start_dim=2, end_dim=-1).permute(0, 2, 1)
        od_test = torch.matmul(concentration, stain_matrix)
        psnr_stain_separation = psnr(od_real, od_test)
        tester.assertTrue(psnr_stain_separation > TestFunctional.THRESH_PSNR,
                          msg=f"{psnr_stain_separation} vs. {TestFunctional.THRESH_PSNR} \n"
                              f"{conc_solver.cfg}")
        dummy_scaled = convert_image_dtype(dummy_tensor_ubyte, torch.float32)
        psnr_out = psnr(dummy_scaled, reconstructed).item()
        tester.assertTrue(psnr_out > TestFunctional.THRESH_PSNR,
                          msg=f"{psnr_out} vs. {TestFunctional.THRESH_PSNR}. \n"
                              f"{get_stain_mat.stain_algorithm.cfg} \n"
                              f"nan: {torch.isnan(reconstructed).any()} \n"
                              f"Dict pos: {CONFIG.DICT_POSITIVE_DICTIONARY}")
        # size
        batch_size, channel_size, height, width = dummy_tensor_ubyte.shape
        tester.assertTrue(stain_matrix.shape == (batch_size, get_stain_mat.num_stains, channel_size))


    def eval_wrapper(self, extractor):
        dummy_tensor_ubyte_list = [TestFunctional.new_dummy_img_tensor_ubyte()
                                   for _ in range(3)]
        # get_stain_mat = MacenkoExtractor()
        dummy_tensor_ubyte = torch.cat(dummy_tensor_ubyte_list, dim=0)
        masks = [torch.ones_like(dummy_tensor_ubyte)[:, 0:1, ...]]
        # all pixel
        # 'ista', 'cd', 'ls', 'fista',
        algorithms = ['ista', 'cd', 'ls', 'fista', 'pinv', 'qr']
        dict_constraint_flag = [True]
        vectorize_flag = [True, False]
        for flag in dict_constraint_flag:
            CONFIG.DICT_POSITIVE_DICTIONARY = flag
            for vf in vectorize_flag:
                CONFIG.ENABLE_VECTORIZE = vf
                for alg in algorithms:
                    for m in masks:
                        cfg = TestFunctional.POSITIVE_CONC_CFG
                        cfg.algorithm = cast(METHOD_FACTORIZE, alg)
                        cfg.positive = True
                        solver = ConcentrationSolver(cfg)
                        TestFunctional.extract_eval_helper(self,
                                                           dummy_tensor_ubyte,
                                                           extractor, luminosity_threshold=None,
                                                           mask=m,
                                                            conc_solver=solver, rng=None)
                        solver.cfg.positive = False
                        TestFunctional.extract_eval_helper(self,
                                                           dummy_tensor_ubyte,
                                                           extractor, luminosity_threshold=None,
                                                           mask=m,
                                                           conc_solver=solver, rng=None)

    def test_stains(self):
        macenko = StainExtraction(MacenkoAlg(DEFAULT_MACENKO_CONFIG), num_stains=2, rng=None)
        vahadane = StainExtraction(VahadaneAlg(DEFAULT_VAHADANE_CONFIG), num_stains=2, rng=None)
        # not support num_stains other than 2
        dummy_tensor = TestFunctional.new_dummy_img_tensor_ubyte()
        with self.assertRaises(AssertionError):
            m3 = StainExtraction(MacenkoAlg(DEFAULT_MACENKO_CONFIG), num_stains=3, rng=None)
            TestFunctional.extract_eval_helper(self,
                                               dummy_tensor,
                                               m3,
                                               conc_solver=ConcentrationSolver(TestFunctional.POSITIVE_CONC_CFG),
                                               luminosity_threshold=None,
                                               mask=None,
                                               rng=None)

        self.eval_wrapper(macenko)
        self.eval_wrapper(vahadane)

        # github remote end fails due to driver issues. Test it locally.
        # # vahadane with rng and lr
        # vahadane.stain_algorithm.cfg.lr = 0.5
        v3 = StainExtraction(VahadaneAlg(DEFAULT_VAHADANE_CONFIG), num_stains=3, rng=None)
        TestFunctional.extract_eval_helper(self, dummy_tensor, v3,
                                           mask=None,
                                           conc_solver=ConcentrationSolver(TestFunctional.POSITIVE_CONC_CFG),
                                           luminosity_threshold=None,
                                           rng=torch.Generator(1))


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

        with self.assertRaises(AssertionError):
            get_tissue_mask(dummy_scaled, luminosity_threshold=0.8,
                            mask = torch.zeros_like(dummy_scaled),
                            throw_error=True)

        with self.assertRaises(TissueMaskException):
            get_tissue_mask(dummy_scaled, luminosity_threshold=0.8,
                            mask = torch.zeros_like(dummy_scaled)[:, 0:1, ...],
                            throw_error=True)
        mask_one = torch.ones_like(dummy_scaled)[:, 0:1, ...]
        assert (mask_one == get_tissue_mask(dummy_scaled, luminosity_threshold=0.8,
                        mask = torch.zeros_like(dummy_scaled)[:, 0:1, ...],
                        throw_error=False, true_when_empty=True)).all()

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

