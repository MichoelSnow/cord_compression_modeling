from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    required = req_file.read().splitlines()

setup(
    name='cord_comp',
    version='0.0.0.0.0.0.0.3',
    description="A computer vision framework to model cord compression",
    author="Glen Ferguson and Michoel Snow",
    author_email='msnow1@montefiore.org',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
)