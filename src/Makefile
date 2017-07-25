PROJECT_ROOT=..
# Handle environment variables
PKG_CONFIG_PATH = /usr/local/opt/opencv3/lib/pkgconfig

CXX   = g++

# ================
# linux
# CXXFLAGS += -c -Wall $(shell pkg-config --cflags opencv) -std=c++11

# OSX
CXXFLAGS += -c -Wall $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags opencv) -std=c++11 -stdlib=libc++
# ================


ODIR  = .obj/build${D}
SDIR  = .
MKDIR = mkdir -p
MV    = mv
LN    = ln -s
BASE_NAME = e2tagging
NAME      = ${BASE_NAME}${D}
OUTDIR    = ${PROJECT_ROOT}/bin

INC       = -I${PROJECT_ROOT}/include -I/usr/local/include
LIBS      = $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs opencv) -L/usr/local/lib -lboost_filesystem -lboost_system

SRC       = main.cpp e2config.cpp
_OBJ      = $(SRC:.cpp=.o)
OBJ       = $(patsubst %,$(ODIR)/%,$(_OBJ))


# Master rule
.PHONY: all
all: ${NAME}


# Output binary
${NAME}: ${OBJ}
	${CXX} ${OBJ} -o ${NAME} ${LIBS} ${LDFLAGS}
	-@${MV} ${NAME} ${OUTDIR}/${NAME}
	-@${LN} ${OUTDIR}/${NAME}

# Intermediate object files
${OBJ}: ${ODIR}/%.o : ${SDIR}/%.cpp
	@${MKDIR} ${ODIR}
	${CXX} ${CXXFLAGS} ${INC} -Wall -c $< -o $@

# Cleanup intermediate objects
.PHONY: clean_obj
clean_obj:
	rm -f ${OBJ}
	@echo "obj cleaned up!"

# Cleanup everything
.PHONY: clean
clean: clean_obj
	rm -f ${OUTDIR}/${NAME} ${OBJ}
	rm -f ${NAME}
	@echo "all cleaned up!"





