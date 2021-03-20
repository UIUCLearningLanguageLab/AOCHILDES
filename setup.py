from setuptools import setup

from aochildes import __name__, __version__

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
    url='https://github.com/UIUCLearningLanguageLab/AOCHILDES',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Retrieve text from the American-English CHILDES database'
)
