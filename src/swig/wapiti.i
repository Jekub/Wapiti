%module(directors="2") Wapiti

%{
#include "ioline.hh"
%}

%include "std_string.i"

%feature("director") IOLine;
%feature("director") WapitiModel;

%include "ioline.hh"

