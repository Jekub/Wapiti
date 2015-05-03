rem rd /s/q fr\limsi\wapiti
rem mkdir fr\limsi\wapiti
rem "%SWIG_HOME%\swig.exe" -java -c++ -package fr.limsi.wapiti -outdir fr\limsi\wapiti wapiti.i

pushd ..\..\build
c:\bin\ninja
popd

"%JAVA_HOME%\bin\javac.exe" -source 1.8 -target 1.8 fr\limsi\wapiti\*.java
"%JAVA_HOME%\bin\jar.exe" cfv wapiti-1.5.0.jar fr\limsi\wapiti\*.class

cp ..\..\build\libwapiti.dll c:\temp\grobid\home\lib\win-64\
cp ..\..\build\libwapiti.dll c:\dev\Grobid.NET\packages\wapiti
cp ..\..\build\libwapiti.dll c:\dev\java-wapiti

rem cp wapiti-1.5.0.jar c:\dev.ext\github-pdf-header-extract\java
cp wapiti-1.5.0.jar c:\dev\grobid.git\grobid-core\deps
cp wapiti-1.5.0.jar C:\dev\grobid.git\grobid-core\lib\fr\limsi\wapiti\wapiti\1.5.0\wapiti-1.5.0.jar
cp wapiti-1.5.0.jar C:\dev\grobid.git\lib\fr\limsi\wapiti\wapiti\1.5.0\wapiti-1.5.0.jar
rem cp wapiti-1.5.0.jar C:\Users\boumenot\.m2\repository\fr\limsi\wapiti\wapiti\1.5.0\wapiti-1.5.0.jar!\fr\limsi\wapiti\wapiti-1.5.0.jar




