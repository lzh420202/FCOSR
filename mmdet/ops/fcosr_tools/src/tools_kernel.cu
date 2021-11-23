#include <ATen/ATen.h>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
#include <numeric>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

const float eps = 1e-9;
const float PI = 3.1415926535898;

#define maxn 51

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif

__device__ inline int sig(const float &d)
{
    return (d > eps) - (d < -eps);
}

__device__ inline int point_eq(const float2 &a, const float2 &b)
{
    return sig(a.x - b.x) == 0 && sig(a.y - b.y) == 0;
}

__device__ inline void point_swap(float2 *a, float2 *b)
{
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2 *last)
{
    while ((first != last) && (first != --last))
    {
        point_swap(first, last);
        ++first;
    }
}

__device__ inline float cross(const float2 &o, const float2 &a, const float2 &b)
{ //叉积
    return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}
__device__ inline float area(float2 *ps, const int &n)
{
    ps[n] = ps[0];
    float res = 0;
    for (int i = 0; i < n; i++)
    {
        res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
    }
    return res / 2.0;
}
__device__ inline int lineCross(const float2 &a, const float2 &b, const float2 &c, const float2 &d, float2 &p)
{
    float s1, s2;
    s1 = cross(a, b, c);
    s2 = cross(a, b, d);
    if (sig(s1) == 0 && sig(s2) == 0)
        return 2;
    if (sig(s2 - s1) == 0)
        return 0;
    p.x = (c.x * s2 - d.x * s1) / (s2 - s1);
    p.y = (c.y * s2 - d.y * s1) / (s2 - s1);
    return 1;
}
__device__ inline void polygon_cut(float2 *p, int &n, const float2 &a, const float2 &b, float2 *pp)
{

    int m = 0;
    p[n] = p[0];
    for (int i = 0; i < n; i++)
    {
        if (sig(cross(a, b, p[i])) > 0)
            pp[m++] = p[i];
        if (sig(cross(a, b, p[i])) != sig(cross(a, b, p[i + 1])))
            lineCross(a, b, p[i], p[i + 1], pp[m++]);
    }
    n = 0;
    for (int i = 0; i < m; i++)
        if (!i || !(point_eq(pp[i], pp[i - 1])))
            p[n++] = pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while (n > 1 && point_eq(p[n - 1], p[0]))
        n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a, float2 b, float2 c, float2 d)
{
    float2 o = make_float2(0, 0);
    int s1 = sig(cross(o, a, b));
    int s2 = sig(cross(o, c, d));
    if (s1 == 0 || s2 == 0)
        return 0.0; //退化，面积为0
    // if (s1 == -1)
    //     point_swap(&a, &b);
    if (s2 == -1)
        point_swap(&c, &d);
    float2 p[10] = {o, a, b};

    if (s1 == -1)
    {
        p[1] = b;
        p[2] = a;
    }
    int n = 3;
    float2 pp[maxn];
    polygon_cut(p, n, o, c, pp);
    polygon_cut(p, n, c, d, pp);
    polygon_cut(p, n, d, o, pp);

    float res = fabs(area(p, n));
    if (s1 * s2 == -1)
        res = -res;
    return res;
}
//求两多边形的交面积
__device__ inline float intersectArea(float2 *ps1, const int &n1, float2 *ps2, const int &n2)
{
    if (area(ps1, n1) < 0)
        point_reverse(ps1, ps1 + n1);
    if (area(ps2, n2) < 0)
        point_reverse(ps2, ps2 + n2);
    ps1[n1] = ps1[0];
    ps2[n2] = ps2[0];
    float res = 0;
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            res += intersectArea(ps1[i], ps1[i + 1], ps2[j], ps2[j + 1]);
        }
    }
    return res; //assumeresispositive!
}

__device__ inline float devPolyIoU(float const *const p, float const *const q)
{
    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++)
    {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    return inter_area / (union_area + eps);
}

__global__ void computePolyIoU(
        const int nthreads,
        float const *poly_1,
        float const *poly_2,
        float *iou)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int offset = index * 8;
        iou[index] = devPolyIoU(poly_1 + offset, poly_2 + offset);
    }
}

at::Tensor tools_compute_poly_iou_cuda(const at::Tensor &poly_1, const at::Tensor &poly_2)
{
    const int dim = poly_1.size(0);
    AT_CHECK(poly_1.sizes() == poly_2.sizes(), "Input 2 polygon must have same shape.");
    AT_CHECK(poly_1.device().index() == poly_2.device().index(), "Input 2 polygon must in same cuda device.");
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, poly_1.device().index());
    auto output = torch::zeros({dim}, options);
    cudaSetDevice(poly_1.device().index());
    computePolyIoU<<<block_count, thread_per_block>>>(
            total_count, poly_1.data<float>(), poly_2.data<float>(), output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

__device__ void
rbox2corners(const float &cx, const float &cy, const float &w, const float &h, const float &angle, float *corners)
{
    /*using opencv format
     * angle (-90, 0]*/
    const float w_x = 0.5 * w * cos(angle);
    const float w_y = 0.5 * w * sin(angle);
    const float h_x = -0.5 * h * sin(angle);
    const float h_y = 0.5 * h * cos(angle);
    corners[0] = cx + w_x + h_x;
    corners[1] = cy + w_y + h_y;

    corners[2] = cx + w_x - h_x;
    corners[3] = cy + w_y - h_y;

    corners[4] = cx - w_x - h_x;
    corners[5] = cy - w_y - h_y;

    corners[6] = cx - w_x + h_x;
    corners[7] = cy - w_y + h_y;
}

__device__ void
rbox2rects(const float &cx, const float &cy, const float &w, const float &h, const float &angle, float *grid_data) {
    /*using opencv format
     * angle (-90.0, 0]
     * */
    float corners[8] = {0};
    rbox2corners(cx, cy, w, h, angle, corners);
    grid_data[0] = min(corners[0], min(corners[2], min(corners[4], corners[6])));
    grid_data[1] = min(corners[1], min(corners[3], min(corners[5], corners[7])));
    grid_data[2] = max(corners[0], max(corners[2], max(corners[4], corners[6])));
    grid_data[3] = max(corners[1], max(corners[3], max(corners[5], corners[7])));
}

/* get_inside_mask */
__device__ int is_inrbox(const float &cx, const float &cy, const float &w, const float &h, const float &angle, const float &x, const float &y)
{
    const float dx = cx - x;
    const float dy = cy - y;
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);
    const float dw = fabs(dx * cos_a + dy * sin_a);
    const float dh = fabs(dy * cos_a - dx * sin_a);
    if((dw <= w * 0.5) & (dh <= h * 0.5))
        return 1;
    else
        return 0;
}

__global__ void get_inside_mask(
        const int nthreads,
        const float *rbox,
        const float *x,
        const float *y,
        unsigned char *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        rbox += index * 5;
        if(is_inrbox(rbox[0], rbox[1], rbox[2], rbox[3], rbox[4], x[index], y[index]) == 1)
            result[index] = 1;
        else
            result[index] = 0;
    }
}

at::Tensor tools_get_inside_mask_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes)
{
    const int dim_1 = rboxes.size(0);
    const int dim_2 = rboxes.size(1);
    AT_CHECK(dim_1 == x.size(0) && dim_2 == x.size(1), "rboxes & x must be same shape");
    AT_CHECK(dim_1 == y.size(0) && dim_2 == y.size(1), "rboxes & y must be same shape");
    const int total_count = dim_1 * dim_2;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    // final output
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim_1, dim_2}, options);
    cudaSetDevice(rboxes.device().index());

    AT_CHECK(rboxes.device().index() == x.device().index(), "rboxes & x must be same device");
    AT_CHECK(rboxes.device().index() == y.device().index(), "rboxes & y must be same device");
    get_inside_mask<<<block_count, thread_per_block>>>(
            total_count, rboxes.data<float>(), x.data<float>(), y.data<float>(), output.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
//    auto result = output.toType(torch::kUInt8);
    return output;
}

__global__ void get_inside_mask_filter(
        const int nthreads,
        const float threshold,
        const float *gds,
        const float *rbox,
        const float *x,
        const float *y,
        unsigned char *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        rbox += index * 5;
        if(is_inrbox(rbox[0], rbox[1], rbox[2], rbox[3], rbox[4], x[index], y[index]) == 1)
        {
            if (gds[index] >= threshold)
                result[index] = 1;
        }
        else
            result[index] = 0;
    }
}

at::Tensor tools_get_inside_mask_with_gds_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes, const float nds_threshold, const at::Tensor &nds)
{
    const int dim_1 = rboxes.size(0);
    const int dim_2 = rboxes.size(1);
    AT_CHECK(dim_1 == x.size(0) && dim_2 == x.size(1), "rboxes & x must be same shape");
    AT_CHECK(dim_1 == y.size(0) && dim_2 == y.size(1), "rboxes & y must be same shape");
    const int total_count = dim_1 * dim_2;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    // final output
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim_1, dim_2}, options);
    cudaSetDevice(rboxes.device().index());

    AT_CHECK(rboxes.device().index() == x.device().index(), "rboxes & x must be same device");
    AT_CHECK(rboxes.device().index() == y.device().index(), "rboxes & y must be same device");
    get_inside_mask_filter<<<block_count, thread_per_block>>>(
            total_count, nds_threshold, nds.data<float>(), rboxes.data<float>(), x.data<float>(), y.data<float>(), output.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
//    auto result = output.toType(torch::kUInt8);
    return output;
}

__global__ void get_inside_mask_with_obj(
        const int nthreads,
        const int dim,
        const float *rbox,
        const float *x,
        const float *y,
        unsigned char *result,
        int *obj_id) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        rbox += index * 5;
        const int id = index % dim;
        if(is_inrbox(rbox[0], rbox[1], rbox[2], rbox[3], rbox[4], x[index], y[index]) == 1) {
            result[index] = 1;
            obj_id[index] = id + 1;
        }
        else
            result[index] = 0;
    }
}

__global__ void get_inside_mask_filter_with_obj(
        const int nthreads,
        const int dim,
        const float threshold,
        const float *gds,
        const float *rbox,
        const float *x,
        const float *y,
        unsigned char *result,
        int *obj_id) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        rbox += index * 5;
        const int id = index % dim;
        if(is_inrbox(rbox[0], rbox[1], rbox[2], rbox[3], rbox[4], x[index], y[index]) == 1)
        {
            if (gds[index] >= threshold) {
                result[index] = 1;
                obj_id[index] = id + 1;
            }
        }
        else
            result[index] = 0;
    }
}

std::vector<at::Tensor> tools_get_inside_mask_with_obj_cuda(
        const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes)
{
    const int dim_1 = rboxes.size(0);
    const int dim_2 = rboxes.size(1);
    AT_CHECK(dim_1 == x.size(0) && dim_2 == x.size(1), "rboxes & x must be same shape");
    AT_CHECK(dim_1 == y.size(0) && dim_2 == y.size(1), "rboxes & y must be same shape");
    const int total_count = dim_1 * dim_2;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    // final output
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, rboxes.device().index());
    auto options_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim_1, dim_2}, options);
    auto idx = torch::zeros({dim_1, dim_2}, options_);
    cudaSetDevice(rboxes.device().index());

    AT_CHECK(rboxes.device().index() == x.device().index(), "rboxes & x must be same device");
    AT_CHECK(rboxes.device().index() == y.device().index(), "rboxes & y must be same device");
    get_inside_mask_with_obj<<<block_count, thread_per_block>>>(
            total_count, dim_2, rboxes.data<float>(),
            x.data<float>(), y.data<float>(),
            output.data<unsigned char>(), idx.data<int>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    //    auto result = output.toType(torch::kUInt8);
    std::vector<at::Tensor> result = {output, idx};
//    result.push_back(output);
//    result.push_back(output);
    return result;
}

std::vector<at::Tensor> tools_get_inside_mask_with_obj_gds_cuda(
        const at::Tensor &x, const at::Tensor &y,
        const at::Tensor &rboxes, const float nds_threshold,
        const at::Tensor &nds)
{
    const int dim_1 = rboxes.size(0);
    const int dim_2 = rboxes.size(1);
    AT_CHECK(dim_1 == x.size(0) && dim_2 == x.size(1), "rboxes & x must be same shape");
    AT_CHECK(dim_1 == y.size(0) && dim_2 == y.size(1), "rboxes & y must be same shape");
    const int total_count = dim_1 * dim_2;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    // final output
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, rboxes.device().index());
    auto options_ = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim_1, dim_2}, options);
    auto idx = torch::zeros({dim_1, dim_2}, options_);
    cudaSetDevice(rboxes.device().index());

    AT_CHECK(rboxes.device().index() == x.device().index(), "rboxes & x must be same device");
    AT_CHECK(rboxes.device().index() == y.device().index(), "rboxes & y must be same device");
    get_inside_mask_filter_with_obj<<<block_count, thread_per_block>>>(
            total_count, dim_2, nds_threshold,
            nds.data<float>(), rboxes.data<float>(),
            x.data<float>(), y.data<float>(),
            output.data<unsigned char>(), idx.data<int>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    //    auto result = output.toType(torch::kUInt8);
    std::vector<at::Tensor> result = {output, idx};
    return result;
}

/* rbox2rect */
__global__ void rbox2rect(
        const int nthreads,
        const float *rbox,
        float *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        const int rbox_idx = index * 5;
        result += index * 4;
        rbox2rects(rbox[rbox_idx], rbox[rbox_idx + 1],
                   rbox[rbox_idx + 2], rbox[rbox_idx + 3],
                   rbox[rbox_idx + 4], result);
    }
}

at::Tensor tools_rbox2rect_cuda(const at::Tensor &rboxes)
{
    const int dim = rboxes.size(0);
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    // final output
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim, 4}, options);
    cudaSetDevice(rboxes.device().index());
    rbox2rect<<<block_count, thread_per_block>>>(
            total_count, rboxes.data<float>(), output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

/* rbox2corner */
__global__ void rbox2corner(
        const int nthreads,
        const float *rbox,
        float *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // locate batch
        const int rbox_idx = index * 5;
        result += index * 8;
        rbox2corners(rbox[rbox_idx], rbox[rbox_idx + 1],
                     rbox[rbox_idx + 2], rbox[rbox_idx + 3],
                     rbox[rbox_idx + 4], result);
    }
}

at::Tensor tools_rbox2corner_cuda(const at::Tensor &rboxes)
{
    const int dim = rboxes.size(0);
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rboxes.device().index());
    auto output = torch::zeros({dim, 8}, options);
    cudaSetDevice(rboxes.device().index());
    rbox2corner<<<block_count, thread_per_block>>>(
            total_count, rboxes.data<float>(), output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

at::Tensor tools_compute_rbox_iou_cuda(const at::Tensor &rbox_1, const at::Tensor &rbox_2)
{
    const int dim = rbox_1.size(0);
    AT_CHECK(rbox_1.sizes() == rbox_2.sizes(), "Input 2 polygon must have same shape.");
    AT_CHECK(rbox_1.device().index() == rbox_2.device().index(), "Input 2 polygon must in same cuda device.");
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rbox_1.device().index());
    auto poly_1 = tools_rbox2corner_cuda(rbox_1);
    auto poly_2 = tools_rbox2corner_cuda(rbox_2);
    auto output = torch::zeros({dim}, options);
    cudaSetDevice(poly_1.device().index());
    computePolyIoU<<<block_count, thread_per_block>>>(
            total_count, poly_1.data<float>(), poly_2.data<float>(), output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

/*compute 2d normal distribution score*/
__device__ void compute_cmi(
        const float &w, const float &h, const float &angle,
        const float &factor, float *result, float *det)
{
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);
    const float sin_a_p2 = sin_a * sin_a;
    const float cos_a_p2 = cos_a * cos_a;
    const float delta_1 = w * w * factor;
    const float delta_2 = h * h * factor;
    const float cm_a = delta_1 * cos_a_p2 + delta_2 * sin_a_p2;
    const float cm_b_c = (delta_1 - delta_2) * sin_a * cos_a;
    const float cm_d = delta_1 * sin_a_p2 + delta_2 * cos_a_p2;
    const float det_ = cm_a * cm_d - cm_b_c * cm_b_c;
    *det = det_;
    if(det_ == 0.0)
        return;
    const float back_det = 1.0 / det_;
    result[0] = cm_d * back_det;
    result[1] = -cm_b_c * back_det;
    result[2] = result[1];
    result[3] = cm_a * back_det;
}

__global__ void get_covariance_matrix_inverse_normal(
        const int nthreads,
        const float factor,
        const float *rbox,
        float *result,
        float *center,
        float *wh,
        float *det
)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        result += index * 4;
        center += index * 2;
        rbox += index * 5;
        wh += index * 2;
        det += index;
        compute_cmi(rbox[2], rbox[3], rbox[4], factor, result, det);
        center[0] = rbox[0];
        center[1] = rbox[1];
        wh[0] = rbox[2];
        wh[1] = rbox[3];
    }
}

__device__ void compute_cmi_shrink(
        const float &w, const float &h, const float &angle,
        const float &factor, float *result, float *wh, float *det) {
    const float sin_a = sin(angle);
    const float cos_a = cos(angle);
    const float sin_a_p2 = sin_a * sin_a;
    const float cos_a_p2 = cos_a * cos_a;
    const float m = min(w, h);
    const float delta_1 = w * m * factor;
    const float delta_2 = h * m * factor;

    const float cm_a = delta_1 * cos_a_p2 + delta_2 * sin_a_p2;
    const float cm_b_c = (delta_1 - delta_2) * sin_a * cos_a;
    const float cm_d = delta_1 * sin_a_p2 + delta_2 * cos_a_p2;
    const float det_ = cm_a * cm_d - cm_b_c * cm_b_c;
    *det = det_;
    if(det_ == 0.0)
        return;
    const float back_det = 1.0 / det_;
    result[0] = cm_d * back_det;
    result[1] = -cm_b_c * back_det;
    result[2] = result[1];
    result[3] = cm_a * back_det;
    wh[0] = sqrt(delta_1 / factor);
    wh[1] = sqrt(delta_2 / factor);
}

__global__ void get_covariance_matrix_inverse_shrink(
        const int nthreads,
        const float factor,
        const float *rbox,
        float *result,
        float *center,
        float *wh,
        float *det) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        result += index * 4;
        center += index * 2;
        rbox += index * 5;
        wh += index * 2;
        det += index;
        compute_cmi_shrink(rbox[2], rbox[3], rbox[4], factor, result, wh, det);
        center[0] = rbox[0];
        center[1] = rbox[1];
    }
}

__global__ void get_normal_distribution_score_kernel(
        const int nthreads,
        const int dim,
        const bool normalize,
        const bool refined,
        const float *wh,
        const float *xs,
        const float *ys,
        const float *center,
        const float *covariance_matrix_inverse,
        const float *covariance_matrix_det,
        float *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int d = index % dim;
        covariance_matrix_inverse += d * 4;
        center += d * 2;
        const float diff_x = xs[index] - center[0];
        const float diff_y = ys[index] - center[1];
        const float value = covariance_matrix_inverse[0] * diff_x * diff_x +
                            (covariance_matrix_inverse[1] + covariance_matrix_inverse[2]) * diff_x * diff_y +
                            covariance_matrix_inverse[3] * diff_y * diff_y;
        if(normalize)
            result[index] = exp(-0.5 * value);
        else
            result[index] = exp(-0.5 * value) / (2.0 * PI * sqrt(covariance_matrix_det[d]) + eps);
            if (refined) {
                wh += d * 2;
                result[index] *= sqrt(wh[0] * wh[1]);
            }
    }
}

at::Tensor tool_normalize_gauss_distribution_score_cuda(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &rboxes, const float &factor,
        const int &mode, const std::vector<int64_t> &block_size)
{
    const int n_points = rboxes.size(0);
    const int n_rboxes = rboxes.size(1);
    const int total_count = n_rboxes;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rboxes.device().index());
    auto covariance_matrix_inverse = torch::zeros({n_rboxes, 4}, options);
    auto covariance_matrix_det = torch::zeros({n_rboxes}, options);
    auto center = torch::zeros({n_rboxes, 2}, options);
    auto wh = torch::zeros({n_rboxes, 2}, options);
    float factor_ = 1.0 / factor;
    AT_CHECK(rboxes.device().index() == covariance_matrix_inverse.device().index(), "rboxes & covariance_matrix must be same device");
    cudaSetDevice(rboxes.device().index());
    if (mode == 0)
    {
        get_covariance_matrix_inverse_normal<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else if (mode == 1)
    {
        get_covariance_matrix_inverse_shrink<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else
        std::cout << "Support mode: Normal(0), Shrink(1)" << std::endl;
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));

    const int main_total_count = n_rboxes * n_points;
    const int main_block_count = (main_total_count + thread_per_block - 1) / thread_per_block;
    auto output = torch::zeros({n_points, n_rboxes}, options);

    bool normalize_flag = true;
    bool refined_flag = false;
    get_normal_distribution_score_kernel<<<main_block_count, thread_per_block>>>(
            main_total_count, n_rboxes, normalize_flag, refined_flag,
            wh.data<float>(), xs.data<float>(), ys.data<float>(),
            center.data<float>(), covariance_matrix_inverse.data<float>(),
            covariance_matrix_det.data<float>(), output.data<float>());
    error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    if(block_size.size() == 1)
    {
        if(block_size[0] == n_points || block_size[0] == -1)
        {
            auto max_value = std::get<0>(torch::max(output, 0));
            output /= max_value;
            return output;
        } else
            std::cout << "Error: block_size must equal to points count when block_size have only one element." << std::endl;
    }
    else if(block_size.size() == 0)
        std::cout << "Error: block_size should contain one element at least." << std::endl;
    else if(std::accumulate(block_size.begin(), block_size.end(), 0) == n_points)
    {

        std::vector<at::Tensor> split_result = output.split_with_sizes(block_size, 0);
        for(int i = 0; i < block_size.size(); i++)
        {
            auto max_value = std::get<0>(split_result[i].max(0));
            split_result[i] /= max_value;
        }
        output = torch::cat(split_result, 0);
        return output;
    } else
        std::cout << "Error: The summation of block_size must equal to total number of points." << std::endl;
    return output;
}

at::Tensor tool_normalize_gauss_distribution_score_cuda_v2(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &rboxes, const float &factor,
        const int &mode)
{
    const int n_points = rboxes.size(0);
    const int n_rboxes = rboxes.size(1);
    const int total_count = n_rboxes;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rboxes.device().index());
    auto covariance_matrix_inverse = torch::zeros({n_rboxes, 4}, options);
    auto covariance_matrix_det = torch::zeros({n_rboxes}, options);
    auto center = torch::zeros({n_rboxes, 2}, options);
    auto wh = torch::zeros({n_rboxes, 2}, options);
    float factor_ = 1.0 / factor;
    AT_CHECK(rboxes.device().index() == covariance_matrix_inverse.device().index(), "rboxes & covariance_matrix must be same device");
    cudaSetDevice(rboxes.device().index());
    if (mode == 0)
    {
        get_covariance_matrix_inverse_normal<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else if (mode == 1)
    {
        get_covariance_matrix_inverse_shrink<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else
        std::cout << "Support mode: Normal(0), Shrink(1)" << std::endl;
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));

    const int main_total_count = n_rboxes * n_points;
    const int main_block_count = (main_total_count + thread_per_block - 1) / thread_per_block;
    auto output = torch::zeros({n_points, n_rboxes}, options);

    bool normalize_flag = true;
    bool refined_flag = false;
    get_normal_distribution_score_kernel<<<main_block_count, thread_per_block>>>(
            main_total_count, n_rboxes, normalize_flag, refined_flag,
            wh.data<float>(), xs.data<float>(), ys.data<float>(),
            center.data<float>(), covariance_matrix_inverse.data<float>(),
            covariance_matrix_det.data<float>(), output.data<float>());
    error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

__global__ void get_max_distance_kernel(
        const int nthreads,
//        const int dim,
        const float *xs,
        const float *ys,
        const float *bboxes,
        float *max_distance) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        bboxes += index * 4;
        const float size_0 = xs[index] - bboxes[0];
        const float size_1 = ys[index] - bboxes[1];
        const float size_2 = bboxes[2] - xs[index];
        const float size_3 = bboxes[3] - ys[index];
        max_distance[index] = max(max(size_0, size_1), max(size_2, size_3));
    }
}

at::Tensor tool_max_distance_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes)
{
    const int n_points = bboxes.size(0);
    const int n_rboxes = bboxes.size(1);
    const int total_count = n_rboxes * n_points;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, bboxes.device().index());
    auto max_distance = torch::zeros({n_points, n_rboxes}, options);

    AT_CHECK(bboxes.device().index() == xs.device().index(), "bboxes & xs must be same device");
    AT_CHECK(bboxes.device().index() == ys.device().index(), "bboxes & ys must be same device");
    cudaSetDevice(bboxes.device().index());
    get_max_distance_kernel<<<block_count, thread_per_block>>>(
            total_count, xs.data<float>(), ys.data<float>(), bboxes.data<float>(), max_distance.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return max_distance;
}

__global__ void get_inside_regress_mask_kernel(
        const int nthreads,
        const float *xs,
        const float *ys,
        const float *bboxes,
        const float *regress_ranges,
        unsigned char *mask) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        bboxes += index * 4;
        regress_ranges += index * 2;
        const float size_0 = xs[index] - bboxes[0];
        const float size_1 = ys[index] - bboxes[1];
        const float size_2 = bboxes[2] - xs[index];
        const float size_3 = bboxes[3] - ys[index];
        const float max_size = max(max(size_0, size_1), max(size_2, size_3));
        if ((size_0 >= 0.0) & (size_1 >= 0.0) & (size_2 >= 0.0) & (size_3 >= 0.0)) {
            if (max_size >= regress_ranges[0] & max_size <= regress_ranges[1])
                mask[index] = 1;
        }
    }
}

at::Tensor tool_inside_regress_mask_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes, const at::Tensor &regress_ranges)
{
    const int n_points = bboxes.size(0);
    const int n_rboxes = bboxes.size(1);
    const int total_count = n_rboxes * n_points;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, bboxes.device().index());
    auto mask = torch::zeros({n_points, n_rboxes}, options);

    AT_CHECK(bboxes.device().index() == xs.device().index(), "bboxes & xs must be same device");
    AT_CHECK(bboxes.device().index() == ys.device().index(), "bboxes & ys must be same device");
    AT_CHECK(bboxes.device().index() == regress_ranges.device().index(), "bboxes & ys must be same device");
    cudaSetDevice(bboxes.device().index());
    get_inside_regress_mask_kernel<<<block_count, thread_per_block>>>(
            total_count, xs.data<float>(), ys.data<float>(), bboxes.data<float>(),
            regress_ranges.data<float>(), mask.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return mask;
}

__global__ void get_inside_balance_regress_mask_kernel(
        const int nthreads,
        const int dim,
        const float *xs,
        const float *ys,
        const float *bboxes,
        const float *rboxes,
        const float *regress_ranges,
        const float *strides,
        const float factor,
        unsigned char *mask) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int points_idx = index / dim;
        bboxes += index * 4;
        rboxes += index * 5;
        regress_ranges += index * 2;
        const float size_0 = xs[index] - bboxes[0];
        const float size_1 = ys[index] - bboxes[1];
        const float size_2 = bboxes[2] - xs[index];
        const float size_3 = bboxes[3] - ys[index];
        const float stride = strides[points_idx];
        const float max_size = max(max(size_0, size_1), max(size_2, size_3));
        const float min_width = min(rboxes[2], rboxes[3]);
        if ((size_0 >= 0.0) & (size_1 >= 0.0) & (size_2 >= 0.0) & (size_3 >= 0.0))
        {
            if (max_size >= regress_ranges[0] & max_size <= regress_ranges[1])
                mask[index] = 1;
            else
            {
                if (max_size >= regress_ranges[1]) {
                    const float valide_size = min_width / (stride * factor);
                    //            if ((valide_size >= 1.0) & (valide_size < 2.0))
                    if (valide_size < 2.0)
                        mask[index] = 1;
                }
            }
        }
    }
}

at::Tensor tool_inside_balance_regress_mask_cuda(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &bboxes, const at::Tensor &rboxes,
        const at::Tensor &regress_ranges, const at::Tensor &strides,
        const float &factor)
{
    const int n_points = bboxes.size(0);
    const int n_rboxes = bboxes.size(1);
    const int total_count = n_rboxes * n_points;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, bboxes.device().index());
    auto mask = torch::zeros({n_points, n_rboxes}, options);

    AT_CHECK(bboxes.device().index() == xs.device().index(), "bboxes & xs must be same device");
    AT_CHECK(bboxes.device().index() == ys.device().index(), "bboxes & ys must be same device");
    AT_CHECK(bboxes.device().index() == regress_ranges.device().index(), "bboxes & ys must be same device");
    cudaSetDevice(bboxes.device().index());
    get_inside_balance_regress_mask_kernel<<<block_count, thread_per_block>>>(
            total_count, n_rboxes, xs.data<float>(), ys.data<float>(), bboxes.data<float>(),
            rboxes.data<float>(), regress_ranges.data<float>(), strides.data<float>(),
            factor, mask.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return mask;
}

__global__ void get_inside_balance_regress_mask_v2_kernel(
        const int nthreads,
        const int dim,
        const float *xs,
        const float *ys,
        const float *rboxes,
        const float *regress_ranges,
        const float *strides,
        const float factor,
        unsigned char *mask) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int points_idx = index / dim;
        rboxes += index * 5;
        regress_ranges += index * 2;

        float bboxes[4] = {0};
        rbox2rects(rboxes[0], rboxes[1], rboxes[2], rboxes[3], rboxes[4], bboxes);

        const float size_0 = xs[index] - bboxes[0];
        const float size_1 = ys[index] - bboxes[1];
        const float size_2 = bboxes[2] - xs[index];
        const float size_3 = bboxes[3] - ys[index];
        const float stride = strides[points_idx];
        const float max_size = max(max(size_0, size_1), max(size_2, size_3));
        const float min_width = min(rboxes[2], rboxes[3]);
        if ((size_0 >= 0.0) & (size_1 >= 0.0) & (size_2 >= 0.0) & (size_3 >= 0.0))
        {
            if (max_size >= regress_ranges[0] & max_size <= regress_ranges[1])
            {
                mask[index] = 1;
            }
            else
            {
                if (max_size >= regress_ranges[1]) {
                    const float valide_size = min_width / (stride * factor);
                    //            if ((valide_size >= 1.0) & (valide_size < 2.0))
                    if (valide_size < 2.0)
                        mask[index] = 1;
                }
            }
        }
    }
}

at::Tensor tool_inside_balance_regress_mask_cuda_v2(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &rboxes, const at::Tensor &regress_ranges,
        const at::Tensor &strides, const float &factor)
{
    const int n_points = rboxes.size(0);
    const int n_rboxes = rboxes.size(1);
    const int total_count = n_rboxes * n_points;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, rboxes.device().index());
    auto mask = torch::zeros({n_points, n_rboxes}, options);

    AT_CHECK(rboxes.device().index() == xs.device().index(), "rboxes & xs must be same device");
    AT_CHECK(rboxes.device().index() == ys.device().index(), "rboxes & ys must be same device");
    AT_CHECK(rboxes.device().index() == regress_ranges.device().index(), "rboxes & ys must be same device");
    cudaSetDevice(rboxes.device().index());
    get_inside_balance_regress_mask_v2_kernel<<<block_count, thread_per_block>>>(
            total_count, n_rboxes, xs.data<float>(), ys.data<float>(), rboxes.data<float>(),
            regress_ranges.data<float>(), strides.data<float>(), factor, mask.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return mask;
}


at::Tensor tool_gauss_distribution_score_cuda(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &rboxes, const float &factor,
        const int &mode, bool refined_flag)
{
    const int n_points = rboxes.size(0);
    const int n_rboxes = rboxes.size(1);
    const int total_count = n_rboxes;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, rboxes.device().index());
    auto covariance_matrix_inverse = torch::zeros({n_rboxes, 4}, options);
    auto covariance_matrix_det = torch::zeros({n_rboxes}, options);
    auto center = torch::zeros({n_rboxes, 2}, options);
    auto wh = torch::zeros({n_rboxes, 2}, options);
    float factor_ = 1.0 / factor;
    AT_CHECK(rboxes.device().index() == covariance_matrix_inverse.device().index(), "rboxes & covariance_matrix must be same device");
    cudaSetDevice(rboxes.device().index());
    if (mode == 0)
    {
        get_covariance_matrix_inverse_normal<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else if (mode == 1)
    {
        get_covariance_matrix_inverse_shrink<<<block_count, thread_per_block>>>(
                total_count, factor_,
                rboxes.data<float>(),
                covariance_matrix_inverse.data<float>(),
                center.data<float>(),
                wh.data<float>(),
                covariance_matrix_det.data<float>());
    } else
        std::cout << "Support mode: Normal(0), Shrink(1)" << std::endl;

    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));

    const int main_total_count = n_rboxes * n_points;
    const int main_block_count = (main_total_count + thread_per_block - 1) / thread_per_block;
    auto output = torch::zeros({n_points, n_rboxes}, options);
    bool normalize_flag = false;
//    bool refined_flag = false;
    get_normal_distribution_score_kernel<<<main_block_count, thread_per_block>>>(
            main_total_count, n_rboxes, normalize_flag, refined_flag,
            wh.data<float>(), xs.data<float>(), ys.data<float>(), center.data<float>(),
            covariance_matrix_inverse.data<float>(), covariance_matrix_det.data<float>(), output.data<float>());
    error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

/*expand_score*/
__global__ void get_expand_score_kernel(
        const int nthreads,
        const int dim,
        const int *box_ids,
        const float *score,
        float *expand_score) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int idx = box_ids[index] - 1;
        if (idx >= 0)
        {
            const int expand_idx = dim * index + idx;
            expand_score[expand_idx] = score[index];
        }
    }
}

at::Tensor tools_get_expand_score_cuda(const at::Tensor &bboxes_ids, const at::Tensor &score, const int num_gt, const float &filled_value)
{
    const int n_points = bboxes_ids.size(0);
    const int total_count = n_points;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, score.device().index());
    auto output = filled_value * torch::ones({n_points, num_gt}, options);

    AT_CHECK(score.device().index() == output.device().index(), "score & output must be same device");
    cudaSetDevice(score.device().index());
    get_expand_score_kernel<<<block_count, thread_per_block>>>(
            total_count, num_gt, bboxes_ids.data<int>(),
            score.data<float>(), output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

/*get keep sample kernel*/
__global__ void get_keep_sample_kernel(
        const int nthreads,
        const int dim,
        const int *dynanic_k,
        const int64_t * topk_ids,
        unsigned char *mask) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        const int index_1 = index / dim;
        const int index_2 = index % dim;
        if (index_1 < dynanic_k[index_2]) {
            const int64_t mask_index = topk_ids[index];
            mask[mask_index] = 1;
        }
    }
}

at::Tensor tools_get_keep_sample_idx_cuda(
        const at::Tensor &dynanic_k,
        const at::Tensor &topk_ids,
        const int &n_points)
{
    auto main_device = topk_ids.device().index();
    AT_CHECK(main_device == dynanic_k.device().index(), "topk_ids & dynanic_k must be same device");
    const int dim_1 = topk_ids.size(0);
    const int dim_2 = topk_ids.size(1);
    const int total_count = dim_1 * dim_2;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, main_device);
    auto output = torch::zeros({n_points}, options);

    AT_CHECK(main_device == output.device().index(), "topk_ids & mask must be same device");
    cudaSetDevice(main_device);
    get_keep_sample_kernel<<<block_count, thread_per_block>>>(
            total_count, dim_2,
            dynanic_k.data<int>(), topk_ids.data<int64_t>(),
            output.data<unsigned char>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}

__device__ inline void get_poly_length(const float2 *polys, float *length, const int n)
{
    /* get length of every edge in polygon*/
    for (int i = 0; i < n; i++)
    {
        float dx = polys[i].x - polys[(i + 1) % n].x;
        float dy = polys[i].y - polys[(i + 1) % n].y;
        length[i] = sqrt(dx * dx + dy * dy);
    }
}

__device__ void poly5to4(const float2 *poly, float2 *out)
{
    /* This function is a cuda kernel implement of DOTA_devkit.ImgSplit_multi_process.splitbase.GetPoly4FromPoly5 */
    float length[5];
    get_poly_length(poly, length, 5);
    float min_v = length[0];
    int min_idx = 0;
    for (int i = 0; i < 5; i++)
    {
        if (length[i] < min_v)
        {
            min_v = length[i];
            min_idx = i;
        }
    }
    int count = 0;
    int out_pos = 0;
    while (count < 5)
    {
        if (count == min_idx)
        {
            out[out_pos].x = (poly[count].x + poly[(count + 1) % 5].x) * 0.5;
            out[out_pos].y = (poly[count].y + poly[(count + 1) % 5].y) * 0.5;
            out_pos += 1;
            count += 1;
        } else if (count == (min_idx + 1) % 5)
        {
            count += 1;
            continue;
        } else
        {
            out[out_pos] = poly[count];
            out_pos += 1;
            count += 1;
        }
    }
}

__global__ void poly_cut_kernel(
        const int nthreads,
        const float *polys,
        const int width,
        const int height,
        const float half_iou_thre,
        const int half_iou_flag,
        float *result) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int n = 4;
        float2 poly[8];
        float2 pp[8];
        float2 box[4] = {make_float2(0, 0),
                         make_float2((int)width - 1, 0),
                         make_float2((int)width - 1, (int)height - 1),
                         make_float2(0, (int)height - 1)};
        polys += index * 8;
        float ori_area = 0.0f;
        for (int i = 0; i < 4; i++)
        {
            poly[i].x = polys[i * 2];
            poly[i].y = polys[i * 2 + 1];
        }
        if (half_iou_flag == 1)
            ori_area = fabs(area(poly, 4));
        polygon_cut(poly, n, box[0], box[1], pp);
        polygon_cut(poly, n, box[1], box[2], pp);
        polygon_cut(poly, n, box[2], box[3], pp);
        polygon_cut(poly, n, box[3], box[0], pp);
        result += index * 8;
        if (n == 4)
        {
            for (int i = 0; i < 4; i++)
            {
                result[i * 2] = poly[i].x;
                result[i * 2 + 1] = poly[i].y;
            }
        }
        else if (n == 5)
        {
            float2 _result[4];
            poly5to4(poly, _result);
            for (int i = 0; i < 4; i++)
            {
                result[i * 2] = _result[i].x;
                result[i * 2 + 1] = _result[i].y;
            }
        } else
        {
            int flag_ = 0;
            if (half_iou_flag == 1)
            {
                if (ori_area > eps)
                {
                    float half_iou = fabs(area(poly, n) / ori_area);
                    if (half_iou > half_iou_thre)
                        flag_ = 1;
                }
            }
            if (flag_ == 1)
            {
                for (int j = 0; j < 8; j++)
                    result[j] = polys[j];
            }
            else
            {
                for (int j = 0; j < 8; j++)
                    result[j] = -1.0;
            }
        }
    }
}

at::Tensor tools_poly_cut_cuda(const at::Tensor &polys, const std::vector<int64_t> &image_size)
{
    auto main_device = polys.device().index();
    const int dim = polys.size(0);
    AT_CHECK(polys.size(1) == 8, "Only support 4-point-polygon.");
    AT_CHECK(image_size.size() == 2, "image size should be a pair number.");
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, main_device);
    auto output = torch::zeros({dim, 8}, options);

    AT_CHECK(main_device == output.device().index(), "new polygon & origen polygon must be same device");
    cudaSetDevice(main_device);
    int width = static_cast<int>(image_size[0]);
    int height = static_cast<int>(image_size[1]);
    poly_cut_kernel<<<block_count, thread_per_block>>>(total_count, polys.data<float>(), width, height, 0.0, 0, output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}


at::Tensor tools_poly_cut_v2_cuda(const at::Tensor &polys, const std::vector<int64_t> &image_size, const float half_iou_thre)
{
    auto main_device = polys.device().index();
    const int dim = polys.size(0);
    AT_CHECK(polys.size(1) == 8, "Only support 4-point-polygon.");
    AT_CHECK(image_size.size() == 2, "image size should be a pair number.");
    const int total_count = dim;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, main_device);
    auto output = torch::zeros({dim, 8}, options);

    AT_CHECK(main_device == output.device().index(), "new polygon & origen polygon must be same device");
    cudaSetDevice(main_device);
    int width = static_cast<int>(image_size[0]);
    int height = static_cast<int>(image_size[1]);
    poly_cut_kernel<<<block_count, thread_per_block>>>(total_count, polys.data<float>(), width, height, half_iou_thre, 1, output.data<float>());
    auto error_code = cudaGetLastError();
    AT_CHECK(error_code == cudaSuccess, cudaGetErrorString(error_code));
    return output;
}