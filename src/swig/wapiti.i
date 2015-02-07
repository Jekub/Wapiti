%module(directors="2") Wapiti

%{
#include "ioline.hh"
%}

%include "std_string.i"

%feature("director") WapitiIO;
%feature("director") WapitiModel;

%include "ioline.hh"

