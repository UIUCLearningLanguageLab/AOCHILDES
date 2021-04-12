from setuptools import setup, find_packages

from aochildes import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=find_packages(include=[__name__, 'original_transcripts']),
    include_package_data=True,
    install_requires=[
        'pandas',
        'pyprind',
    ],
    url='https://github.com/UIUCLearningLanguageLab/AOCHILDES',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Retrieve text from the American-English CHILDES database'
)
