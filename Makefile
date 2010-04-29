CC     =cc
CFLAGS =-std=c99 -W -Wall -O3
LIBS   =-lm -lpthread

ARCH   =-march=native

DESTDIR=
PREFIX =/usr/local

INSTALL= install -p
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644

wapiti: src/wapiti.c
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -DNDEBUG $(CFLAGS) $(ARCH) $(LIBS) -o wapiti src/wapiti.c

debug: src/wapiti.c
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -g $(CFLAGS) $(ARCH) $(LIBS) -o wapiti src/wapiti.c

install: wapiti
	@echo "CP: wapiti   --> $(DESTDIR)$(PREFIX)/bin"
	@mkdir -p $(DESTDIR)$(PREFIX)/bin
	@mkdir -p $(DESTDIR)$(PREFIX)/man/man1
	@$(INSTALL_EXEC) wapiti       $(DESTDIR)$(PREFIX)/bin
	@$(INSTALL_DATA) doc/wapiti.1 $(DESTDIR)$(PREFIX)/share/man/man1

clean:
	@echo "RM: wapiti"
	@rm -f wapiti

.PHONY: clean install
