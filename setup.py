from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    required = req_file.read().splitlines()

print(required)

setup(
    name='cv_prototyping_framework',
    version='0.0.0.0.0.0.0.3',
    description="A computer vision framework for rapidly prototyping models",
    author="Glen Ferguson and Michoel Snow",
    author_email='glen@ferguson76.com',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
)