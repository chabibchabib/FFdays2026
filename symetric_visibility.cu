// nvcc -ptx -arch=sm_86 -I<OptiX>/include BuildVisibilityPairs.cu -o BuildVisibilityPairs.ptx
#include <optix.h>
#include <cuda_runtime.h>
#include "params.h"

extern "C" __constant__ Params params;

__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); } // Definition of - Operator 
__device__ __forceinline__ float  dot3(float3 a, float3 b)      { return a.x*b.x + a.y*b.y + a.z*b.z; } // Scalar product

//Indexing function  gives the start idx of row i 
__device__ __forceinline__ unsigned long long rowOffset(unsigned long long i, unsigned long long N) {
    return i * (2ull * N - i - 1ull) / 2ull;
}

// from a global idx k returns  (i,j)  the coordinates of the Th.nbe  grid where  i<j
__device__ __forceinline__ void unrank(unsigned long long k, unsigned long long N,
                                       unsigned &i, unsigned &j)
{
    double b = 2.0 * (double)N - 1.0;
    double disc = b * b - 8.0 * (double)k;
    if (disc < 0.0) disc = 0.0;
    long long ii = (long long)((b - sqrt(disc)) * 0.5);
    if (ii < 0) ii = 0;
    while (ii > 0 && rowOffset(ii, N) > k)        --ii;   // corrections
    while (rowOffset(ii + 1, N) <= k)             ++ii;
    i = (unsigned)ii;
    j = (unsigned)(k - rowOffset(ii, N) + ii + 1);
}

extern "C" __global__ void __raygen__rg()
{
    const unsigned long long k = params.k0 + optixGetLaunchIndex().x;
    if (k >= params.numPairs) return;

    unsigned a, b;
    // Fill a & b from the global idx k
    unrank(k, params.N, a, b);

    const float3 P = params.pts[a]; 
    const float3 Q = params.pts[b];
    float3 d = Q - P; // direction of the ray
    const float L = sqrtf(dot3(d, d));
    if (L <= 0.f) return;
    float Linv=1./L;
    d.x *= Linv; d.y *= Linv; d.z *= Linv; // Normalize d

    //const float tol = 1e-3f;
    //if (dot3(params.nin[a], d) <=  tol) return;
    //if (dot3(params.nin[b], d) >= -tol) return;

    unsigned occluded = 1;   // pessimiste ; le miss le passe a 0
    optixTrace(params.gas, P, d,
               params.tminRel * L, params.tmaxRel * L, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 1, 0,
               occluded);
    // Storage 
    if (!occluded)
        atomicOr(&params.bits[k >> 5], 1u << (k & 31));
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);    // rien touche -> visible
}
