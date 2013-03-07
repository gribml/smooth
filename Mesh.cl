/*
bool isSurfaceNode(size_t vid, **** *NEList, **** *NNList) {
  return NEList[3*vid].size() < NNList[vid].size();
}
*/
bool isCornerNode(size_t vid, float* normals) {
  fabs(normals[2*vid])==1.0 && fabs(normals[2*vid+1]==1.0);
}
/*
float element_area(size_t eid, **** *ENList, float* coords, int orientation) {
  const int3 n = ENList[3*eid];
  const float c0 = coords[2*n[0]];
  const float c1 = coords[2*n[1]];
  const float c2 = coords[2*n[2]];

  return 0.5 * orientation * ((c0[1] - c2.y) * (c0[0] - c1[0]) -
                              (c0[1] - c1.y) * (c0[0] - c2[0]));

}


void element_quality(size_t eid, int3* ENList, 
                     float* metric, float* coords) {
  const int3 n = ENList[3*eid];
  
  const float c0 = &coords[2*n[0]];
  const float c1 = &coords[2*n[1]];
  const float c2 = &coords[2*n[2]];

  const float m0 = &metric[2*n[0]];
  const float m1 = &metric[2*n[1]];
  const float m2 = &metric[2*n[2]];

  // Metric tensor averaged over the element
  float m00 = (m0[0] + m1[0] + m2[0])/3;
  float m01 = (m0[1] + m1[1] + m2[1])/3;
  float m11 = (m0[2] + m1[2] + m2[2])/3;

  // l is the length of the perimeter, measured in metric space
  float l =
    sqrt((c0[1] - c1[1])*((c0[1] - c1[1])*m11 + (c0[0] - c1[0])*m01) +
         (c0[0] - c1[0])*((c0[1] - c1[1])*m01 + (c0[0] - c1[0])*m00))+
    sqrt((c0[1] - c2[1])*((c0[1] - c2[1])*m11 + (c0[0] - c2[0])*m01) +
         (c0[0] - c2[0])*((c0[1] - c2[1])*m01 + (c0[0] - c2[0])*m00))+
    sqrt((c2[1] - c1[1])*((c2[1] - c1[1])*m11 + (c2[0] - c1[0])*m01) +
         (c2[0] - c1[0])*((c2[1] - c1[1])*m01 + (c2[0] - c1[0])*m00));

  // Area in physical space
  float a = element_area(eid);

  // Area in metric space
  float a_m = a*sqrt(m00*m11 - m01*m01);

  // Function
  float f = l < 3.0 ? l/3.0 : 3.0/l;
  float F = f * (2.0 - f);
  F = F * F * F;
  return 12.0 * sqrt(3) * a_m * F / (l*l);
}

__kernel void smooth
(__global float *metric, __global float *coords,
 __global ****** NEList, __global ****** *NNList,
 __global float *normals, __global size_t *ENList,
 int orientation, size_t NElements, size_t NNodes) {*/
(/*__global float3 *metric, */__global float *coords,
 __global float *normals, const uint NNodes) {
  int vid = get_global_id(0);

  if(vid >= NNodes)
    return;

  if(isCornerNode(vid, normals))
    return;

  /*
  float worst_q = 1.0;

  int i;
  float qual;
  for( i = 0; i < NEList[vid].numElements; ++i ) {
    qual = element_quality();
    worst_q = min( worst_q, qual );
  }
  
  const float m0 = metric[3*vid];
  float x0 = coords[2*vid];
  float y0 = coords[2*vid+1];

//  float4 A = {0.0, 0.0, 0.0, 0.0};
//  float2 q = {0.0, 0.0};
  double A[4] = {0.0, 0.0, 0.0, 0.0};
  double q[2] = {0.0, 0.0};
  
   for ( i = 0; i < NNList[vid].numElements; ++i ) {
    
    size_t il = NNList[vid].index;
 
//    const float3 m1 = metric[il];
    const float m1 = metric[3*il];
   
//    float ml00 = 0.5*(m0.x + m1.x);
//    float ml01 = 0.5*(m0.y + m1.y);
//    float ml11 = 0.5*(m0.z + m0.z);

    float ml00 = 0.5*(m0[0] + m1[0]);
    float ml01 = 0.5*(m0[1] + m1[1]);
    float ml11 = 0.5*(m0[2] + m0[2]);
   
//    float2 coord1 = coords[il];
    float x = 
//    q.x += ( ml00*coord1.x + ml01*coord1.y);
//    q.y += ( ml01*coord1.x + ml11*coord1.y);
//
//    A.x += ml00;
//    A.y += ml01;
//    A.w += ml11;
   
  }
  
  A.z = A.y;
  float2 p;

  p.y = (q.y - A.z/A.x) / (A.w - A.y*A.z/A.x);
  p.x = (q.x - A.x*p.y) / A.x;
  
  if ( isSurfaceNode(vid) ) {
    p.x -= p.x * fabs( normals[vid].x );
    p.y -= p.y * fabs( normals[vid].y );
  }

  coords[vid].x += p.x;
  coords[vid].y += p.y;

  float new_worst_q = 1.0;
  for ( i = 0; i < NEList[vid].numElements; ++i ) {
    qual = element_quality();
    new_worst_q = min( new_worst_q, qual );
  }

  if ( new_worst_q < worst_q ) {
    coords[vid].x -= p.x;
    coords[vid].y -= p.y;
    }
}
  
