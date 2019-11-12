from setuptools import setup

from src import __name__, __version__

setup(
    name='ccc',
    version=__version__,
    packages=[__name__],
    include_package_data=True,
    install_requires=['cached_property',
                      'pandas',
                      'scipy',
                      'numpy',
                      'sklearn',
                      'spacy',
                      'pyprind',
                      'sortedcontainers',
                      'matplotlib',
                      'seaborn',
                      'cytoolz',
                      'PyYAML',
                      'attrs'],
    url='https://github.com/UIUCLearningLanguageLab/CreateCHILDESCorpus',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Create text corpus from American-English CHILDES database'
)
