from setuptools import setup

setup(
    name='CHILDESHub',
    version='0.1dev',
    packages=['childeshub', 'analysis'],
    install_requires=['cached_property',
                      'pandas',
                      'scipy',
                      'numpy',
                      'sklearn',
                      'spacy',
                      'pyprind',
                      'sortedcontainers', 'matplotlib', 'seaborn'],
    url='https://github.com/phueb/CHILDESHub',
    license='',
    author='Philip Huebner',
    author_email='',
    description='Create and analyze variants of American-English CHILDES corpus'
)


