#!/usr/bin/env python

from setuptools import (
        setup as install,
        find_packages,
        )

VERSION = '0.0.1'

install(
        name='drl',
        packages=['drl'],
        version=VERSION,
        description='Distributed reinforcement learning algorithms implemented in Pytorch.',
        author='Seb Arnold',
        author_email='smr.arnold@gmail.com',
        url='https://github.com/seba-1511/drl.pth',
        download_url='https://github.com/seba-1511/drl/archive/0.1.3.zip',
        license='License :: OSI Approved :: Apache Software License',
        classifiers=[],
        scripts=[]
)
