from setuptools import setup, find_packages
from pkg_resources import parse_requirements
with open('requirements.txt') as root:
    requirements = [str(req) for req in parse_requirements(root)]

version_dict = {}
with open("./torch_staintools/version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]

setup(
    name='torch-staintools',
    version=version,
    packages=find_packages(),
    url='',
    license='MIT',
    author='Y Z',
    author_email='cielmercy@gmail.com',
    description='GPU-accelerated stain normalization as nn.Module.',
    install_requires=requirements,
)
