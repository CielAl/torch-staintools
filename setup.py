from setuptools import setup, find_packages
from pkg_resources import parse_requirements
with open('requirements.txt') as root:
    requirements = [str(req) for req in parse_requirements(root)]

version_dict = {}
with open("./torch_staintools/version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torch-staintools',
    version=version,
    packages=find_packages(exclude=["tests*", "tests.*"]),
    url='https://github.com/CielAl/torch-staintools',
    license='MIT',
    author='Y Z',
    author_email='cielmercy@gmail.com',
    description='GPU-accelerated stain normalization and augmentation',
    install_requires=requirements,
    python_requires='>=3.10',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
