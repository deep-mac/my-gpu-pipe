#include "blockgemm.cuh"
#include "utils.cuh"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/util/host_tensor.h"

#define NI 100000
#define NW 8
#define M 64
#define N 128
#define K 32

// using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;

using Block = cutlass::gemm::GemmShape<M, N, K>;

using BlockGemmOp = ThreadBlockGemmTensorOp<NW, Block>;

template <typename Mma, typename ThreadblockShape>
__global__ void kernel3(
  typename Mma::ElementC *output_C, 
  typename Mma::ElementA const *input_A,
  typename Mma::ElementB const *input_B,
  typename Mma::ElementC const *input_C,
  int iterations = NI) {

  // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

  __shared__ cutlass::AlignedBuffer<
    typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

  if (threadIdx.x == 0) {
    typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
    #pragma unroll 1
    for (int i = 0; i < smem_buffer_A.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementA>::type>::get(input_A, i);
    }

    typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
    #pragma unroll 1
    for (int i = 0; i < smem_buffer_B.size(); ++i) {
      cutlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
          cutlass::ReferenceFactory<typename cutlass::platform::remove_const<
              typename Mma::ElementB>::type>::get(input_B, i);
    }
  }

  __syncthreads();

  //
  // Construct warp-level matrix product
  //

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;

  typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
  typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
  typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});

  //typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());

  //typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());

  FragmentA frag_A;
  FragmentB frag_B;

  FragmentC accum;

  Mma mma;

  accum.clear();

  CUTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {     // place in loop that is not unrolled 
    typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, cutlass::arch::LaneId());

    typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, cutlass::arch::LaneId());


    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK;
         k += Mma::Policy::MmaShape::kK) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);

      ++iter_A;
      ++iter_B;

      mma(accum, frag_A, frag_B, accum);
    }
  }
  
  typename Mma::IteratorC iter_C({output_C, layout_C}, cutlass::arch::LaneId());

  iter_C.store(accum);
}

__global__ void kernel() {
    BlockGemmOp gemm_op;
    __shared__ half I[M * K];
    __shared__ half W[N * K];
    __shared__ half O[M * N];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    for (int i = 0; i < NI; i++) {
        gemm_op(
            (cutlass::half_t *)I,
            (cutlass::half_t *)W,
            (cutlass::half_t *)O,
            warp_id,
            lane_id);
    }

}
using WarpShape = cutlass::gemm::GemmShape<M, N / NW, K>;

using WarpGemmOp = cutlass::gemm::warp::GemmTensorOp<
    WarpShape,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<cutlass::half_t>::value, WarpShape::kK>,
    cutlass::layout::RowMajor,
    true
    >;

__global__ void kernel2() {
    WarpGemmOp gemm_op;
    __shared__ half I[M * K];
    __shared__ half W[N * K / NW];
    __shared__ half O[M * N];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    for (int i = 0; i < NI; i++) {
        gemm_op(
            WarpGemmOp::TensorRefA((cutlass::half_t *)I, K),
            WarpGemmOp::TensorRefB((cutlass::half_t *)W, K),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N / NW),
            WarpGemmOp::TensorRefC((cutlass::half_t *)O, N / NW),
            lane_id);
    }

}

int main() {
    dim3 grid(1, 1);
    dim3 block(32, NW);

    //half * I;
    //cudaErrCheck(cudaMalloc(&I, M * K * sizeof(half)));


    /*printf("Running...\n");
    float time_ms = cuda_time_kernel_ms(
        [&]() {
            kernel<<<grid, block>>>();
        }
    );

    printf("gemm1 took %fms\n", time_ms);

    float flops_v1 = 2.0f * M * N * K * NI;
    float gflops_v1 = flops_v1 / (time_ms * 1e6);
    printf("+ GFLOPS: %f\n", gflops_v1);


     printf("Running...\n");
     float time_ms2 = cuda_time_kernel_ms(
         [&]() {
             kernel2<<<grid, block>>>();
         }
     );

     printf("gemm2 took %fms\n", time_ms2);

     float flops_v2 = 2.0f * M * N * K * NI;
     float gflops_v2 = flops_v2 / (time_ms2 * 1e6);
     printf("+ GFLOPS: %f\n", gflops_v2);*/
    using Shape = cutlass::gemm::GemmShape<M, N, K>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using Element = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, 32>;
    using LayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, 32>;

    using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
        Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
        cutlass::layout::RowMajor>::Type;
    using Mma = MmaTensorOp;
    using ThreadblockShape = cutlass::gemm::GemmShape<M*4, N*2, K>;
    using Operator = cutlass::arch::OpMultiplyAdd
        ;

    using Shape = typename Mma::Shape;
    using ElementA = typename Mma::ElementA;
    using LayoutA = typename Mma::LayoutA;
    using ElementB = typename Mma::ElementB;
    using LayoutB = typename Mma::LayoutB;
    using ElementC = typename Mma::ElementC;
    using LayoutC = typename Mma::LayoutC;

    cutlass::HostTensor<ElementA, LayoutA> tensor_A;
    cutlass::HostTensor<ElementB, LayoutB> tensor_B;
    cutlass::HostTensor<ElementC, LayoutC> tensor_C;
    cutlass::HostTensor<ElementC, LayoutC> tensor_D_computed;

    tensor_A.reset(cutlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(cutlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(cutlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(cutlass::make_Coord(Shape::kM, Shape::kN));

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();

  // launch kernel
  printf("Running...\n");
  float time_ms3 = cuda_time_kernel_ms(
    [&]() {
        kernel3<Mma, ThreadblockShape><<< dim3(1, 1), dim3(32, NW, 1) >>>(
          tensor_D_computed.device_data(),
          tensor_A.device_data(),
          tensor_B.device_data(),
          tensor_C.device_data());});


     /*float time_ms3 = cuda_time_kernel_ms(
         [&]() {
             kernel3<<<grid, block>>>();
         }
     );*/

     printf("gemm3 took %fms\n", time_ms3);

     float flops_v3 = 2.0f * M * N * K * NI*NW;
     float gflops_v3 = flops_v3 / (time_ms3 * 1e6);
     printf("+ GFLOPS: %f\n", gflops_v3);

    return 0;
}
