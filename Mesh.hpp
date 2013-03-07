//============================================================================
// Name        : Mesh.hpp
// Author      : George Rokos
// Description : Mesh description
//============================================================================

#ifndef MESH_HPP_
#define MESH_HPP_

#include <cstddef>
#include <set>
#include <vector>

struct Quality{
  double mean;
  double min;
  double rms;
};

class Mesh{
public:
  // Constructor
  Mesh(const char *filename);

  size_t NNodes;    // Number of mesh vertices.
  size_t NElements; // Number of mesh elements.

  // Element eid is comprised of the vertices
  // ENList[3*eid], ENList[3*eid+1] and ENList[3*eid+2].
  std::vector<size_t> ENList;

  // Vertex vid has coordinates x=coords[2*vid] and y=coords[2*vid+1].
  std::vector<double> coords;

  // The metric tensor at vertex vid is M_00 = metric[3*vid],
  //                                    M_01 = M_10 = metric[3*vid+1] and
  //                                    M_11 = metric[3*vid+2].
  std::vector<double> metric;

  /* If vid is on the surface, the normal vector
   * (normals[2*vid],normals[2*vid+1] =
   *                            = (0.0,1.0) if vid is on the top surface
   *                            = (0.0,-1.0) if vid is on the bottom surface
   *                            = (1.0,0.0) if vid is on the right surface
   *                            = (-1.0,0.0) if vid is on the left surface
   * For all other vertices, the normal vector is (0.0,0.0).
   */
  std::vector<double> normals;

  // For every vertex i, NNList[i] contains the IDs of all adjacent vertices.
  std::vector< std::vector<size_t> > NNList;

  // For every vertex i, NEList[i] contains the IDs of all adjacent elements.
  std::vector< std::set<size_t> > NEList;

  bool isSurfaceNode(size_t vid) const;
  bool isCornerNode(size_t vid) const;
  double element_area(size_t eid) const;
  double element_quality(size_t eid) const;
  Quality get_mesh_quality() const;

private:
  void create_adjacency();
  void find_surface();
  void set_orientation();

  int orientation;
};

#endif /* MESH_HPP_ */
