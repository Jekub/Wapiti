cmake_minimum_required(VERSION 2.8)
project( wapiti )

set(CMAKE_C_FLAGS "-std=c99")

find_package(JNI REQUIRED)

if (JNI_FOUND)
    include_directories( ${JNI_INCLUDE_DIRS} )
endif (JNI_FOUND)

AUX_SOURCE_DIRECTORY(src SOURCES)

AUX_SOURCE_DIRECTORY(src/swig SOURCES_SWIG)
if(APPLE)
elseif(UNIX)
  list(REMOVE_ITEM SOURCES_SWIG src/swig/fmemopen.c)
endif()
message ("Swig sources list: ${SOURCES_SWIG}")

#SET_SOURCE_FILES_PROPERTIES( ${SOURCES} PROPERTIES LANGUAGE CXX )

add_library( wapiti SHARED ${SOURCES} ${SOURCES_SWIG})

set( CMAKE_VERBOSE_MAKEFILE on )