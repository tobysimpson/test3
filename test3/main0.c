//
//  main.c
//  test3
//
//  Created by Toby Simpson on 06.02.24.
//

#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <OpenCL/opencl.h>


//trying block sparse, unions on host - copied from test13_solve3.c in test1

#define ROOT_WRITE  "/Users/toby/Downloads/"

union flt4
{
    cl_float4  vec;
    float      arr[4];

};


union flt4x4
{
    cl_float16  vec;
    float       arr[4][4];
};


//write
void wrt_raw(void *ptr, size_t n, size_t bytes, char *file_name)
{
//    printf("%s\n",file_name);
    
    //name
    char file_path[250];
    sprintf(file_path, "%s%s.raw", ROOT_WRITE, file_name);

    //open
    FILE* file = fopen(file_path,"wb");
  
    //write
    fwrite(ptr, bytes, n, file);
    
    //close
    fclose(file);
    
    return;
}


//sparse assemble then write to blas then change row data
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
    //vec
    union flt4 uu[4];
    union flt4 bb[4];
    
    DenseVector_Float u = {16, (float*)uu};
    DenseVector_Float b = {16, (float*)bb};

    //coo
    int             ii[4] = {0,1,2,3};
    int             jj[4] = {0,1,2,3};
    union flt4x4  vv[4];

    
    //init
    for(int k=0; k<4; k++)
    {
        for(int j=0; j<4; j++)
        {
            uu[k].arr[j] = 4*k+j;
            bb[k].arr[j] = 0e0f;
            
            for(int i=0; i<4; i++)
            {
                vv[k].arr[j][i] = 4*i+j;        //col maj
                vv[k].arr[j][i] = 2*(i==j);     //col maj
            }
        }
    }
    
    //write vec
    wrt_raw(uu, 4, sizeof(cl_float4),  "uu");
    wrt_raw(bb, 4, sizeof(cl_float4),  "bb");
    
    //write mtx
    wrt_raw(ii, 4, sizeof(int),         "A_ii");
    wrt_raw(jj, 4, sizeof(int),         "A_jj");
    wrt_raw(vv, 4, sizeof(cl_float16),  "A_vv");
    
    SparseAttributes_t atts;
    atts.kind = SparseOrdinary;
    
    long    blk_num = 4;
    UInt8   blk_sz  = 4;
    
    int n = 4;
    
    SparseMatrix_Float A = SparseConvertFromCoordinate(n, n, blk_num, blk_sz, atts, ii, jj, (float*)vv);  //duplicates sum
    
    
    SparseMultiply(A,u,b);
    
    memset(uu,0, 4*sizeof(cl_float4));
    
    SparseSolve(SparseGMRES(), A, b, u);
    SparseCleanup(A);

    //write vec
    wrt_raw(uu, 4, sizeof(cl_float4),  "uu");
    wrt_raw(bb, 4, sizeof(cl_float4),  "bb");
    
    
    printf("done\n");
    
    return 0;
}

