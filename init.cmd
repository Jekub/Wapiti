rem Works on my machine...
rm -rf CMakeCache.txt CMakeFiles

rem call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64
rem set PATH=%PATH%;c:\Program Files\LLVM\bin
rem set PATH=%PATH%;c:\Program Files (x86)\Windows Kits\10\bin\10.0.15063.0\x64

"c:\Program Files\CMake\bin\cmake.exe"^
  -G "Visual Studio 15 2017 Win64"^
  -T LLVM-vs2014^
  .
