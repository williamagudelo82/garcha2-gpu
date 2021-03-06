#-*- mode: make -*-

# sources and objects
MOD_SRCS=$(wildcard mods/*.f)
COMMON_SRCS=$(wildcard *.f)
GARCHA_SRCS=$(wildcard garcha_src/*.f)

MOD_OBJ=$(MOD_SRCS:%.f=%.o)
GARCHA_OBJ=$(MOD_OBJ) $(COMMON_SRCS:%.f=%.o) $(GARCHA_SRCS:%.f=%.o)

# flags
ifeq ($(profile),1)
  PROFILE  = -pg
else
  PROFILE  =
endif

DEFINE   = -Dpack
CFLAGS   = -g -mp1 -ip -O3 -c -cpp $(DEFINE) $(PROFILE)
CFLAGS2  = -g -O3 -c -cpp $(DEFINE) $(PROFILE)
CFLAGS3  = -g -O1 -c -cpp $(DEFINE) $(PROFILE)
pPPFLAGS = -P -traditional-cpp
LIBS=-L/usr/lib -L/usr/lib64 -L../g2g -lg2g -lstdc++ -L$(MKLROOT)/lib/em64t $(PROFILE) -lmkl_solver_lp64 -lmkl_core \
     -lmkl_intel_thread -liomp5 -lmkl_lapack -lpthread -lm -lmkl_intel_lp64 -Wl,-rpath='$$ORIGIN/' -Wl,-rpath='$$ORIGIN/../g2g'
# directories
G2G_CFLAGS=-DG2G

ifeq ($(ultima_garcha),1)
	G2G_CFLAGS += -DULTIMA_GARCHA
endif

ifeq ($(convergencia_garcha),1)
	G2G_CFLAGS += -DINT3N_GARCHA
endif

ifeq ($(ultima_g2g),1)
	CFLAGS += -DULTIMA_G2G
	CFLAGS2 += -DULTIMA_G2G
endif

## Targets: make ##

all: dirs garcha-g2g garcha

dirs:
	mkdir -p obj/garcha_g2g_obj/mods obj/garcha_g2g_obj/garcha_src
	mkdir -p obj/garcha_obj/mods obj/garcha_obj/garcha_src

garcha-g2g: dirs $(GARCHA_OBJ:%.o=obj/garcha_g2g_obj/%.o)
	ifort -o $@ $(GARCHA_OBJ:%.o=obj/garcha_g2g_obj/%.o) $(LIBS)

garcha: dirs $(GARCHA_OBJ:%.o=obj/garcha_obj/%.o)
	ifort -o $@ $(GARCHA_OBJ:%.o=obj/garcha_obj/%.o) $(LIBS)


## Targets: clean ##
clean:
	rm -rf obj
	rm -rf garcha garcha-g2g


# Different flags for some .o
obj/garcha_g2g_obj/dip2.o: dip2.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/dip2.o: dip2.f
	ifort -o $@ $(CFLAGS3) $<

obj/garcha_g2g_obj/dipG.o: dipG.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/dipG.o: dipG.f
	ifort -o $@ $(CFLAGS3) $<

obj/garcha_g2g_obj/garcha_src/dip.o: garcha_src/dip.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/garcha_src/dip.o: garcha_src/dip.f
	ifort -o $@ $(CFLAGS3) $<

obj/garcha_g2g_obj/garcha_src/int1G.o: garcha_src/int1G.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/garcha_src/int1G.o: garcha_src/int1G.f
	ifort -o $@ $(CFLAGS3) $<

obj/garcha_g2g_obj/garcha_src/int3G.o: garcha_src/int3G.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/garcha_src/int3G.o: garcha_src/int3G.f
	ifort -o $@ $(CFLAGS3) $<

obj/garcha_g2g_obj/garcha_src/intsolG.o: garcha_src/intsolG.f
	ifort -o $@ $(CFLAGS3) $(G2G_CFLAGS) $<
obj/garcha_obj/garcha_src/intsolG.o: garcha_src/intsolG.f
	ifort -o $@ $(CFLAGS3) $<

### Objects
obj/garcha_obj/%.o: %.f
	ifort -o $@ $(CFLAGS) $<

obj/garcha_g2g_obj/%.o: %.f
	ifort -o $@ $(CFLAGS) $(G2G_CFLAGS) $<
