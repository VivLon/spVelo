from setuptools import setup, find_packages

setup(
    name='spVelo',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.8',
    author='Wenxin Long',
    author_email='wbl5283@psu.edu',
    description='RNA velocity inference for multi-batch spatial transcriptomics dataset.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/VivLon/spVelo',
)