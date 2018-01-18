#define SWAP(a,b) {__local float * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global float * input, __global float * output, __local float * a, __local float * b)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);
 
    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    output[gid] = a[lid];
}

__kernel void aggregate_sums(__global float * input, __global float * output, __global float * sums)
{
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);
    output[gid] = input[gid] + sums[gid / block_size];
}
