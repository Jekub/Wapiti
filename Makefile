CC     =cc
CFLAGS =-std=c99 -W -Wall -O3
LIBS   =-lm -lpthread

ARCH   =-march=native

DESTDIR=
PREFIX =/usr/local

INSTALL= install -d
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644

wapiti: src/wapiti.c
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -DNDEBUG $(CFLAGS) $(LIBS) -o wapiti src/wapiti.c

install: wapiti
	@echo "CP: wapiti   --> $(DESTDIR)$(PREFIX)/bin"
	@$(INSTALL_EXEC) $(DESTDIR)$(PREFIX)/bin wapiti

clean:
	@echo "RM: wapiti"
	@rm -f wapiti

.PHONY: clean install
