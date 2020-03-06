INCDIR = -I/usr/include/opencv
DBG    = -g -pg
OPT    = -O3
CPP    = g++
CFLAGS = -Wall $(INCDIR) -Wno-sign-compare

LIBDIR = -L/usr/lib
LINK   = -lopencv_core -lopencv_highgui -lopencv_imgproc

SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard *.h)



.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: program

program: $(SOURCES) $(HEADERS)
	$(CPP) $(OPT) $(CFLAGS) -o streamFCSS $(SOURCES) $(LIBDIR) $(LINK)

debug: $(SOURCES) $(HEADERS)
	$(CPP) $(DBG) $(CFLAGS) -o streamFCSS_debug $(SOURCES) $(LIBDIR) $(LINK)

clean:
	/bin/rm -f streamFCSS streamFCSS_debug *.o

clean-all: clean
	/bin/rm -f *~
