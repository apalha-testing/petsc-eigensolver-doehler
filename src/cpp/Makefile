# include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
#
#
# eigensolver_doehler: eigensolver_doehler.o JDMaxEigenvalue.o
# 	-${CLINKER} -g -o main main.o  JDMaxEigenvalue.o -lm -ldl ${SLEPC_EPS_LIB}
# 	${RM} *.o
#
# JDMaxEigenvalue.o: JDMaxEigenvalue.cpp


TARGET      = doehler_eigensolver
BUILD_DIR   = src
SRC_DIR     = src

default: $(TARGET)

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

OBJ =  \
doehler_eigensolver_main.o\
doehler_eigensolver_kernel.o\
doehler_eigensolver_inout.o


OBJ_D       = $(patsubst %,$(BUILD_DIR)/%,$(OBJ))

dir_guard=@mkdir -p $(@D)


$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# $(BUILD_DIR)/%.o:          $(SRC_DIR)/%.cpp
# 	$(CXX) $(CXXFLAGS) $(IFLAGS) $(LFLAGS) -c -o $@ $<

$(TARGET): $(BUILD_DIR) $(OBJ_D)
	$(dir_guard)
	-${CLINKER} -mmacosx-version-min=12.2 -g -o $(TARGET) $(OBJ_D) ${SLEPC_EPS_LIB}
	${RM} $(BUILD_DIR)/*.o

test_slepc: src/test_slepc.o
	-${CLINKER} -mmacosx-version-min=12.2 -g -o test_slepc src/test_slepc.o ${SLEPC_EPS_LIB}
	${RM} src/*.o
