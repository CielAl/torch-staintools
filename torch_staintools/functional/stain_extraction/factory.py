from .vahadane import VahadaneAlg
from .macenko import MacenkoAlg
from .extractor import StainExtraction


def build_from_name(algo: str) -> StainExtraction:
    """A factory builder to create stain extractor from name.

    Args:
        algo: support 'macenko' and 'vahadane'

    Returns:

    """
    algo = algo.lower()
    match algo:
        case 'macenko':
            return MacenkoAlg()
        case 'vahadane':
            return VahadaneAlg()
    raise ValueError(f"{algo} not defined")
