from .vahadane import VahadaneExtractor
from .macenko import MacenkoExtractor
from .extractor import BaseExtractor


def build_from_name(algo: str) -> BaseExtractor:
    algo = algo.lower()
    match algo:
        case 'macenko':
            return MacenkoExtractor()
        case 'vahadane':
            return VahadaneExtractor()
    raise ValueError(f"{algo} not defined")
