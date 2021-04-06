from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='zviz',
    packages=['zviz'],

    version='0.9.0',

    license='MIT',

    install_requires=['networkx'],

    author='hokuseihal',
    author_email='hokuseihal@gmail.com',

    url='https://github.com/hokuseihal/zviz',

    description='Visualize the progress of your torch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='zviz',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)