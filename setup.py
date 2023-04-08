from setuptools import setup,find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='W2VBiLSTM',
    version='0.0.1',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
)