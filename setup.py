from setuptools import setup

setup(
    name='CHILDESHub',
    version='0.1dev',
    packages=['childeshub'],
    include_package_data=True,
    install_requires=['cached_property',
                      'pandas',
                      'scipy',
                      'numpy',
                      'sklearn',
                      'spacy',
                      'pyprind',
                      'sortedcontainers', 'matplotlib', 'seaborn', 'cytoolz'],
    url='https://github.com/phueb/CHILDESHub',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Create and analyze variants of American-English CHILDES corpus'
)