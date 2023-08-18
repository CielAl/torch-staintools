from setuptools import setup
from pkg_resources import parse_requirements
with open('requirements.txt') as root:
    requirements = [str(req) for req in parse_requirements(root)]

setup(
    name='torch_stain_tools',
    version='0.0.1',
    packages=['torch_stain_tools', 'torch_stain_tools.functional', 'torch_stain_tools.functional.conversion',
              'torch_stain_tools.functional.tissue_mask', 'torch_stain_tools.functional.optimization',
              'torch_stain_tools.functional.preprocessing', 'torch_stain_tools.functional.stain_extraction',
              'torch_stain_tools.normalizer'],
    url='',
    license='MIT',
    author='YZ',
    author_email='cielmercy@gmail.com',
    description='',
    install_requires=requirements,
)
