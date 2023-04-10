#pragma once

#include "warpgemm.cuh"


template <
    size_t NWARPS,
    typename ThreadBlockShape>
class ThreadBlockGemmTensorOp
{
public:
    using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;
    static const size_t K = ThreadBlockShape::kK;

    using WarpGemmOp = cutlass::gemm::warp::GemmTensorOp<
        WarpShape,
        cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
            cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
        cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
            cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
        cutlass::layout::RowMajor,
        true
        >;

    static constexpr size_t nblk_m = ThreadBlockShape::kM / WarpShape::kM;
    static constexpr size_t blk_m = WarpShape::kM;

    static constexpr size_t nblk_n = ThreadBlockShape::kN / WarpShape::kN;
    static constexpr size_t blk_n = WarpShape::kN;

    static constexpr size_t nblk = nblk_m * nblk_n;

    static constexpr size_t nblk_k = ThreadBlockShape::kK / WarpShape::kK;
    static constexpr size_t blk_k = WarpShape::kK;


    WarpGemmOp warp_gemm_op;

    __device__ void operator()(
            typename WarpGemmOp::MmaWarp::ElementA* A,
            typename WarpGemmOp::MmaWarp::ElementB* B,
            typename WarpGemmOp::MmaWarp::ElementC* C,
            size_t warp_id,
            size_t lane_id
    ) const {

        /*WarpGemmOp::MmaWarp::ElementA* A_base = A;
        WarpGemmOp::MmaWarp::ElementB* B_base = B;

        WarpGemmOp::MmaWarp::ElementC* C_ptr = C;
        WarpGemmOp::MmaWarp::ElementA* A_ptr = A_base;
        WarpGemmOp::MmaWarp::ElementB* B_ptr = B_base;*/
        printf("nblk_m = %ld, blk_m = %ld, nblk_n = %ld, blk_n = %ld, nblk = %ld, nblk_k = %ld, blk_k = %ld\n", nblk_n, blk_m, nblk_n, blk_n, nblk, nblk_k, blk_k);
        #pragma unroll
        for (size_t blk = warp_id; blk < nblk; blk += NWARPS) {
            const size_t bm = blk / nblk_n;
            const size_t m = bm * blk_m;

            const size_t bn = blk % nblk_n;
            const size_t n = bn * blk_n;

            WarpGemmOp::MmaWarp::ElementA* A_base = A + m * K;
            WarpGemmOp::MmaWarp::ElementB* B_base = B + n * K;

            WarpGemmOp::MmaWarp::ElementC* C_ptr = C + m * ThreadBlockShape::kN + n;

            #pragma unroll
            for (size_t kblk = 0; kblk < nblk_k; ++kblk) {
                WarpGemmOp::MmaWarp::ElementA* A_ptr = A_base + kblk * blk_k;
                WarpGemmOp::MmaWarp::ElementB* B_ptr = B_base + kblk * blk_k;

                warp_gemm_op(
                    WarpGemmOp::TensorRefA(A_ptr, K),
                    WarpGemmOp::TensorRefB(B_ptr, K),
                    WarpGemmOp::TensorRefC(C, ThreadBlockShape::kN),
                    WarpGemmOp::TensorRefC(C, ThreadBlockShape::kN),
                    lane_id
                );
            }
        }
    }

};
