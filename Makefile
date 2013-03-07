CXXFLAGS = -O3 -Wall -Wno-deprecated -I/homes/gr409/public_html/vtk/include/vtk-5.10

OBJS = ACA2-2013.o Mesh.o Smooth.o SVD2x2.o

LIBS = -L/homes/gr409/public_html/vtk/lib/vtk-5.10 -lvtkIO -lvtkFiltering -lvtkCommon -lvtkzlib -lvtkexpat -lvtksys -ldl -lpthread

TARGET = ACA2-2013

$(TARGET):	$(OBJS)
	@$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	@$(TARGET)

clean:
	@rm -f $(OBJS) $(TARGET)
