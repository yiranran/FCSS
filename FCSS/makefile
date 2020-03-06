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
	$(CPP) $(OPT) $(CFLAGS) -o FCSS $(SOURCES) $(LIBDIR) $(LINK)

debug: $(SOURCES) $(HEADERS)
	$(CPP) $(DBG) $(CFLAGS) -o FCSS_debug $(SOURCES) $(LIBDIR) $(LINK)

clean:
	/bin/rm -f FCSS FCSS_debug *.o

clean-all: clean
	/bin/rm -f *~
