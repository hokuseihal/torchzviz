from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

import sys
major,minor,_,_,_=sys.version_info
assert major>=3 and minor>=7,"USE UPPER PYTHON 3.7"

url = 'https://github.com/hokuseihal/torchzviz'
setup(
    name='zviz',
    packages=['zviz'],
    package_data={'zviz':['*py','utils/*']},

    version='1.0.4',

    license='MIT',

    install_requires=['networkx','pygraphviz'],

    author='hokuseihal',
    author_email='hokuseihal@gmail.com',

    url='https://github.com/hokuseihal/torchzviz',

    description='Visualize the progress of your torch.',
    long_description=f'See [Here]({url})',
    long_description_content_type='text/markdown',
    keywords='zviz',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)