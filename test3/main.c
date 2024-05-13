//
//  main.c
//  test3
//
//  Created by Toby Simpson on 06.02.24.
//

#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <OpenCL/opencl.h>


//re-learning sparse solvers

void fn_disp(float v[4])
{
    for(int i=0; i<4; i++)
    {
        printf("%f\n",v[i]);
    }
}

void fn_print_csr(SparseMatrix_Float A)
{
    float aa[A.structure.rowCount*A.structure.columnCount];
    
    //reset
    for(int i=0; i<A.structure.rowCount*A.structure.columnCount; i++)
    {
        aa[i] = 0;
    }
    
    
    for(int col_idx=0; col_idx<A.structure.columnCount; col_idx++)
    {
//        printf("%ld %ld\n", A.structure.columnStarts[col_idx], A.structure.columnStarts[col_idx+1]);
        
        for(long row_ptr=A.structure.columnStarts[col_idx]; row_ptr<A.structure.columnStarts[col_idx+1]; row_ptr++)
        {
//            printf("%lu ", row_ptr);
            
            int row_idx = A.structure.rowIndices[row_ptr];
            float   val = A.data[row_ptr];
            
            aa[row_idx*A.structure.columnCount+col_idx] = val;
        }
        
//        printf("\n");
    }
    
    
    
    for(int j=0; j<A.structure.columnCount; j++)
    {
        for(int i=0; i<A.structure.rowCount; i++)
        {
            printf("%f ",aa[j*A.structure.columnCount+i]);
        }
        printf("\n");
    }
    
    
    
}


//sparse assemble then write to blas then change row data
int main(int argc, const char * argv[])
{
    printf("hello\n");
    

    //vec
    float uu[4] = {1,1,1,1};
    float bb[4] = {0,0,0,0};
    
    
    DenseVector_Float u = {4, uu};
    DenseVector_Float b = {4, bb};

    //coo
    int     ii[6] = {0,1,2,3,0,1};
    int     jj[6] = {0,1,2,3,1,0};
    float   vv[6] = {1,1,1,1,2,3};

    
    //init
    SparseAttributes_t atts;
//    atts.kind = SparseOrdinary;
//    atts.kind = SparseSymmetric;            //sums and stores by triangle type
    atts.kind = SparseTriangular;         //discards according to triangle type, default is upper
    
//    atts.triangle = SparseLowerTriangle;
//    atts.triangle = SparseUpperTriangle;
    

    
    long    nnz     = 6;
    UInt8   blk_sz  = 1;
    
    int n = 4;
    
    SparseMatrix_Float A = SparseConvertFromCoordinate(n, n, nnz, blk_sz, atts, ii, jj, vv);  //duplicates sum
    
    //over-ride after init!!!
    A.structure.attributes.kind = SparseSymmetric;
    
    fn_print_csr(A);
    
    //mult
    SparseMultiply(A,u,b);
    fn_disp(b.data);
    
    //reset
    memset(u.data,0, 4*sizeof(float));
//    fn_disp(u.data);
    
    
    
    SparseSolve(SparseGMRES(), A, b, u);
//    SparseSolve(SparseConjugateGradient(), A, b, u);
    SparseCleanup(A);


    fn_disp(u.data);
    
    
    printf("done\n");
    
    return 0;
}

