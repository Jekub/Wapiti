#include <iostream>
#include <string>

class WapitiIO {
public:
    virtual ~WapitiIO() {}
    virtual char *readline() { std::cout << "you called me!\n"; return NULL; }
    virtual void append(char *data) {}
};

extern "C" {
    #include "../decoder.h"
    #include "../ioline.h"
    #include "../model.h"
    #include "../options.h"
    #include "../reader.h"
}

char *wapiti_gets_cb(void *x) {
    WapitiIO *io = static_cast<WapitiIO*>(x);
    char *p = io->readline();

    if (p != NULL) {
        int len = strlen(p);
        char *s = (char*)xmalloc(len + 1);
        strncpy(s, p, len + 1);
        return s;
    }

    return NULL;
}

int wapiti_print_cb(void *x, char *format, ...) {
    WapitiIO *io = static_cast<WapitiIO*>(x);
    
    va_list args;
    va_start(args, format);
    size_t len = vsnprintf(NULL, 0, format, args);
    char *buf = (char*)xmalloc(len + 1);
    len = vsnprintf(buf, len, format, args);
    va_end(args);

    if (len > 0)
        buf[len] = '\0';

    io->append(buf);
    free(buf);

    return len;
}

class WapitiModel {
private:
    WapitiIO *_io;
    iol_t *_iol;
    rdr_t *_rdr;
    mdl_t *_mdl;

public:
    WapitiModel(WapitiIO *io) {
        _io = io;

        _iol = iol_new2(
            wapiti_gets_cb, 
            static_cast<void*>(io), 
            wapiti_print_cb, 
            static_cast<void*>(io));

        _rdr = rdr_new(_iol, opt_defaults.maxent);
        _mdl = mdl_new(_rdr);
        _mdl->opt = &opt_defaults;
        
        mdl_load(_mdl);
    }

    ~WapitiModel() {
        iol_free(_iol);
        rdr_free(_rdr);
        mdl_free(_mdl);
    }

    void label(WapitiIO *io) {
        iol_t *iol = iol_new2(
            wapiti_gets_cb,
            io,
            wapiti_print_cb,
            io);

        tag_label(_mdl, iol);
        iol_free(iol);
    }
};
