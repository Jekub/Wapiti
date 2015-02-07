#include <iostream>
#include <string>

extern "C" {
    #include "../decoder.h"
    #include "../ioline.h"
    #include "../model.h"
    #include "../options.h"
    #include "../reader.h"
}

class IOLine {
public:
    virtual ~IOLine() {}
    virtual char *readline() { return NULL; }
    virtual void append(char *data) {}
};

char *wapiti_gets_cb(void *x) {
    IOLine *ioline = static_cast<IOLine*>(x);
    char *p = ioline->readline();

    if (p != NULL) {
        int len = strlen(p);
        char *s = (char*)xmalloc(len + 1);
        strncpy(s, p, len + 1);
        return s;
    }

    return NULL;
}

int wapiti_print_cb(void *x, char *format, ...) {
    IOLine *ioline = static_cast<IOLine*>(x);
    
    va_list args;
    va_start(args, format);
    size_t len = vsnprintf(NULL, 0, format, args);
    char *buf = (char*)xmalloc(len + 1);
    len = vsnprintf(buf, len, format, args);
    va_end(args);

    if (len > 0)
        buf[len] = '\0';

    ioline->append(buf);
    free(buf);

    return len;
}

class WapitiModel {
private:
    IOLine *_ioline;
    iol_t *_iol;
    rdr_t *_rdr;
    mdl_t *_mdl;

public:
    WapitiModel(IOLine *ioline) {
        _ioline = ioline;

        _iol = iol_new2(
            wapiti_gets_cb, 
            static_cast<void*>(ioline), 
            wapiti_print_cb, 
            static_cast<void*>(ioline));

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

    void label(IOLine *lines) {
        iol_t *iol = iol_new2(
            wapiti_gets_cb,
            lines,
            wapiti_print_cb,
            lines);

        tag_label(_mdl, iol);
        iol_free(iol);
    }
};
