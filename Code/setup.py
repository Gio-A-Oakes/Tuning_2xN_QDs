from setuptools import setup, find_packages

"""
For more information on the package, see:
https://github.com/Gio-A-Oakes/Tuning_DQD
"""

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='Tuning_QDs',
    version='1.0.0',
    url='https://github.com/Gio-A-Oakes/Tuning_DQD',
    license='MIT',
    author='Giovanni Oakes',
    author_email='gioakes@gmail.com',
    description='Systematically tuning a 2×N array of quantum dots with machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Experimentalists working on QD arrays',
        'Topic :: Quantum dots :: Quantum computing :: Machine learning',

        # Pick your license as you wish (should match "license" above)
         'License :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='Quantum dots, Quantum computing, Machine learning',
    python_requires='>=3.6',
)
