# CXXFLAGS = -O3 -Wall -Wno-deprecated -I/homes/gr409/public_html/vtk/include/vtk-5.10 -I.
CXXFLAGS = -O3 -Wall -Wno-deprecated -I/usr/local/Cellar/vtk/5.10.1/include/vtk-5.10 -I.
OBJS = ACA2-2013.o Mesh.o Smooth.o SVD2x2.o CLWrapper.o

# LIBS = -lOpenCL -L/homes/gr409/public_html/vtk/lib/vtk-5.10 -lvtkIO -lvtkFiltering -lvtkCommon -lvtkzlib -lvtkexpat -lvtksys -ldl -lpthread
LIBS = -framework OpenCL -L/usr/local/Cellar/vtk/5.10.1/lib/vtk-5.10 -lvtkIO -lvtkFiltering -lvtkCommon -lvtkzlib -lvtkexpat -lvtksys -ldl -lpthread

TARGET = ACA2-2013

$(TARGET):	$(OBJS)
	@$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	@rm -f $(OBJS) $(TARGET)
