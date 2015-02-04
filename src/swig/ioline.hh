#include <iostream>
#include <string>

class IOLine {
public:
    virtual ~IOLine() {}
    virtual char *readline() { return "mule"; }
    virtual void append(char *data) {}
};

extern "C" {
    #include <string.h>
    #include "../decoder.h"
    #include "../ioline.h"
    #include "../model.h"
    #include "../options.h"
    #include "../reader.h"

    opt_t* get_default_opt() 
    {
        opt_t *opt = new opt_t();

        opt->mode    = -1;
        opt->input   = NULL;     opt->output  = NULL;
        opt->type    = "crf";
        opt->maxent  = false;
        opt->algo    = "l-bfgs"; opt->pattern = NULL;  opt->model   = NULL; opt->devel   = NULL;
        opt->rstate  = NULL;     opt->sstate  = NULL;
        opt->compact = false;    opt->sparse  = false;
        opt->nthread = 1;        opt->jobsize = 64;    opt->maxiter = 0;
        opt->rho1    = 0.5;      opt->rho2    = 0.0001;
        opt->objwin  = 5;        opt->stopwin = 5;     opt->stopeps = 0.02;

        opt->label   = false;    opt->check   = false; opt->outsc = false;
        opt->lblpost = false;    opt->nbest   = 1;     opt->force = false;
        opt->prec    = 5;
        opt->all     = false;

        opt->lbfgs.clip   = false;
        opt->lbfgs.histsz = 5;
        opt->lbfgs.maxls = 40;

        opt->sgdl1.eta0   = 0.8;
        opt->sgdl1.alpha  = 0.85;

        opt->bcd.kappa  = 1.5;

        opt->rprop.stpmin = 1e-8;
        opt->rprop.stpmax = 50.0;
        opt->rprop.stpinc = 1.2;
        opt->rprop.stpdec = 0.5,
        opt->rprop.cutoff = false;

        return opt;
    }
}

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
    opt_t *_opt;
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

        _rdr = rdr_new(_iol, _opt->maxent);
        _mdl = mdl_new(_rdr);

        _opt = get_default_opt();
        _mdl->opt = _opt;
        
        mdl_load(_mdl);
    }

    ~WapitiModel() {
        delete _opt;

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
