# This modified version is modified yet again by Christopher Boumenot.

The modified modified version adds support for Windows.

# This is a modified version of Wapiti 1.5, patched by Vyacheslav Zholudev

The present modified version brings to the latest Wapiti release the following features:

- SWIG mapping providing in particular JNI interface for integrating Wapiti in Java, both for training and decoding,
- CMake, cross-platform build system, instead of Make, allowing to build the library more easily in the environment of your choice.

The source code was patched not to use VLA (i.e. `main(int argc, char*[argc] argv)`. Instead, `main(int argc, char** argv)` is used). 

To build the code, from the Wapiti root path: 

    mkdir build; 
	cd build; 
	cmake ..; 
	make 
	
`libwapiti.so` or `libwapiti.dylib` will appear in the same directory and should not contains dynamic dependencies (for portability of the JNI), which can be checked on Linux by:

    ldd libwapiti.so

or on Darwin architecture:

    otool -L libwapiti.dylib

The jar file can we found under `src/swig`.

# Wapiti - A linear-chain CRF tool

    Copyright (c) 2009-2013  CNRS
    All rights reserved.

For more detailed information see the [homepage](http://wapiti.limsi.fr).

Wapiti is a very fast toolkit for segmenting and labeling sequences with
discriminative models. It is based on maxent models, maximum entropy Markov
models and linear-chain CRF and proposes various optimization and regularization
methods to improve both the computational complexity and the prediction
performance of standard models. Wapiti is ranked first on the sequence tagging
task for more than a year on MLcomp web site.

Wapiti is developed by LIMSI-CNRS and was partially funded by ANR projects
CroTaL (ANR-07-MDCO-003) and MGA (ANR-07-BLAN-0311-02).

For suggestions, comments, or patchs, you can contact me at lavergne@limsi.fr

If you use Wapiti for research purpose, please use the following citation:

    @inproceedings{lavergne2010practical,
        author    = {Lavergne, Thomas and Capp\'{e}, Olivier and Yvon,
                     Fran\c{c}ois},
        title     = {Practical Very Large Scale {CRFs}},
        booktitle = {Proceedings the 48th Annual Meeting of the Association
                     for Computational Linguistics ({ACL})},
        month     = {July},
        year      = {2010},
        location  = {Uppsala, Sweden},
        publisher = {Association for Computational Linguistics},
        pages     = {504--513},
        url       = {http://www.aclweb.org/anthology/P10-1052}
    }

