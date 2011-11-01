CFLAGS =-std=c99 -W -Wall -Wextra -O3
LIBS   =-lm -lpthread

DESTDIR=
PREFIX =/usr/local

INSTALL= install -p
INSTALL_EXEC= $(INSTALL) -m 0755
INSTALL_DATA= $(INSTALL) -m 0644

SRC=src/*.c
HDR=src/*.h

wapiti: $(SRC) $(HDR)
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -DNDEBUG $(CFLAGS) $(LIBS) -o wapiti $(SRC)

debug: $(SRC) $(HDR)
	@echo "CC: wapiti.c --> wapiti"
	@$(CC) -g $(CFLAGS) $(LIBS) -o wapiti $(SRC)

install: wapiti
	@echo "CP: wapiti   --> $(DESTDIR)$(PREFIX)/bin"
	@mkdir -p $(DESTDIR)$(PREFIX)/bin
	@mkdir -p $(DESTDIR)$(PREFIX)/share/man/man1
	@$(INSTALL_EXEC) wapiti       $(DESTDIR)$(PREFIX)/bin
	@$(INSTALL_DATA) doc/wapiti.1 $(DESTDIR)$(PREFIX)/share/man/man1

clean:
	@echo "RM: wapiti"
	@rm -f wapiti

.PHONY: clean install
