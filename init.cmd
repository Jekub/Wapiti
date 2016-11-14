rem Works on my machine...

rm -rf CMakeCache.txt CMakeFiles

rem call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64

"c:\Program Files\CMake\bin\cmake.exe"^
 -DCMAKE_C_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe"^
 -DCMAKE_CXX_COMPILER="c:/Program Files/LLVM/bin/clang-cl.exe"^
 -DCMAKE_BUILD_TYPE=Release^
 -G Ninja^
 .
