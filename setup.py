from setuptools import setup

from childes import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=[__name__],
    include_package_data=True,
    install_requires=['cached_property',
                      'pandas',
                      'numpy',
                      'pyprind',
                      'sortedcontainers',
                      ],
    url='https://github.com/UIUCLearningLanguageLab/CreateCHILDESCorpus',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='API for retrieving text from American-English CHILDES database'
)
