#include "cutlass/gemm/warp/default_mma_tensor_op.h"

using LayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<Element>::value, 64>;

using LayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
    cutlass::sizeof_bits<Element>::value, 64>;

using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<64, 64, 8>,                            // Overall warp-level GEMM operation
    cutlass::gemm::GemmShape<16, 8, 16>,                             // Target instruction
    cutlass::half_t, LayoutA,                                       // operand A type and layout
    cutlass::half_t, LayoutB,                                       // operand B type and layout
    cutlass::half_t,                                                          // accumulator type
    cutlass::layout::RowMajor>::Type;                               // accumulator layout

//
// Define a GEMM operation loading data from shared memory
//
int const kGemmK = 32;

__shared__ ElementA smem_buffer_A[WarpMma::Shape::kM * kGemmK];
__shared__ ElementB smem_buffer_B[WarpMma::Shape::kN * kGemmK];

//
// Construct iterators into SMEM tiles
//

// leading dimensions inferred from matrix problem size
int lda = WarpMma::Shape::kM;
int ldb = WarpMma::Shape::kN;

// iterators into shared memory
WarpMma::IteratorA warp_iterator_A({smem_buffer_A, lda});
WarpMma::IteratorB warp_iterator_B({smem_buffer_B, ldb});

// Fragments in registers storing the operands
FragmentA frag_A;
FragmentB frag_B;
FragmentC accum;

WarpMma mma;

accum.clear();

//
// Accumulated outer product
//

#pragma unroll 1
for (int k = 0; k < kGemmK; k += WarpMma::Shape::kK) {

  
  iter_A.load(frag_A);  // Load fragments from A and B matrices
  iter_B.load(frag_B);

  ++iter_A; ++iter_B;   // Advance along GEMM K to next tile in A
                        //   and B matrices

                        // Compute matrix product
  mma(accum, frag_A, frag_B, accum);
}
