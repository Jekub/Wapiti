[![Build status](https://ci.appveyor.com/api/projects/status/ry6ki3d31qneg5po?svg=true)](https://ci.appveyor.com/project/boumenot/wapiti)

# Wapiti

This is a fork of [Wapiti][wapiti].  Please see the [original
site][wapiti] for the definitive source.

## My Fork

This version has been modified to support:

 * Windows (x64).
 * Streaming IO interface for loading data in leiu of reading files
    off disk.
 * Intentional support for C# via P/Invoke.

This fork builds upon and makes use of other forks.  These forks
provide the following features.

 * CMake, cross-platform build system, instead of Make, allowing to
   build the library more easily in the environment of your choice.
 * Remove most uses of VLA (i.e. `main(int argc, char*[argc]
   argv)`. Instead, `main(int argc, char** argv)` is used).

## Branches

There are two other branches in this repository worthy of note.

 1. [grobid-java-swig-win32](https://github.com/boumenot/wapiti/tree/grobid-java-swig-win32)
 1. [grobid-iolines-swig-win32](https://github.com/boumenot/wapiti/tree/grobid-iolines-swig-win32)

The branch grobid-java-swig-win32 is for the Wapiti version used by
[Grobid][grobid].  This is a Windows build of Wapiti with SWIG binding
for Java.

The branch grobid-iolines-swig-win32 is an experimental version of
Wapiti with a streaming IO interface with SWIG bindings for Java.

[wapiti]: https://github.com/Jekub/Wapiti
[grobid]: https://github.com/kermitt2/grobid
