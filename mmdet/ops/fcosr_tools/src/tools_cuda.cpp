#include <torch/torch.h>
#include <torch/extension.h>
#include <stdio.h>
#include <vector>
// CUDA forward declarations

#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif

at::Tensor tools_compute_poly_iou_cuda(const at::Tensor &poly_1, const at::Tensor &poly_2);

at::Tensor tools_compute_rbox_iou_cuda(const at::Tensor &rbox_1, const at::Tensor &rbox_2);

at::Tensor tools_get_inside_mask_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes);

std::vector<at::Tensor> tools_get_inside_mask_with_obj_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes);

at::Tensor tools_get_inside_mask_with_gds_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes, const float nds_threshold, const at::Tensor &nds);

std::vector<at::Tensor> tools_get_inside_mask_with_obj_gds_cuda(const at::Tensor &x, const at::Tensor &y,
                                                                const at::Tensor &rboxes, const float nds_threshold,
                                                                const at::Tensor &nds);

at::Tensor tools_rbox2rect_cuda(const at::Tensor &rboxes);

at::Tensor tools_rbox2corner_cuda(const at::Tensor &rboxes);

at::Tensor tool_normalize_gauss_distribution_score_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const float &factor,  const int &mode, const std::vector<int64_t> &block_size);

at::Tensor tool_normalize_gauss_distribution_score_cuda_v2(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const float &factor,  const int &mode);

at::Tensor tool_gauss_distribution_score_cuda(const at::Tensor &xs, const at::Tensor &ys,const at::Tensor &rboxes, const float &factor, const int &mode, bool refined);

at::Tensor tool_max_distance_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes);

at::Tensor tool_inside_regress_mask_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes, const at::Tensor &regress_ranges);

at::Tensor tool_inside_balance_regress_mask_cuda(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes, const at::Tensor &rboxes, const at::Tensor &regress_ranges, const at::Tensor &strides, const float &factor);

at::Tensor tool_inside_balance_regress_mask_cuda_v2(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const at::Tensor &regress_ranges, const at::Tensor &strides, const float &factor);

at::Tensor rebalance_weight_cuda(const at::Tensor &weight, const at::Tensor &label, const at::Tensor &idx, const std::vector<int64_t> &block_size);

at::Tensor tools_get_expand_score_cuda(const at::Tensor &bboxes_ids, const at::Tensor &score, const int num_gt, const float &filled_value);

at::Tensor tools_get_keep_sample_idx_cuda(const at::Tensor &dynanic_k, const at::Tensor &topk_ids, const int &n_points);

at::Tensor tools_poly_cut_cuda(const at::Tensor &polys, const std::vector<int64_t> &image_size);

at::Tensor tools_poly_cut_v2_cuda(const at::Tensor &polys, const std::vector<int64_t> &image_size, const float half_iou_thre);
// C++ interface
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor tools_compute_poly_iou(const at::Tensor &poly_1, const at::Tensor &poly_2)
{
    CHECK_INPUT(poly_1);
    CHECK_INPUT(poly_2);
    return tools_compute_poly_iou_cuda(poly_1, poly_2);
}

at::Tensor tools_compute_rbox_iou(const at::Tensor &rbox_1, const at::Tensor &rbox_2)
{
    CHECK_INPUT(rbox_1);
    CHECK_INPUT(rbox_2);
    return tools_compute_rbox_iou_cuda(rbox_1, rbox_2);
}

at::Tensor tools_get_inside_mask(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes)
{
    /* rboxes: Tensor shape[N, 5]
     * */
    CHECK_INPUT(rboxes);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    return tools_get_inside_mask_cuda(x, y, rboxes);
}

at::Tensor tools_rbox2rect(const at::Tensor &rboxes)
{
    /* rboxes: Tensor shape[N, 5]
     * */
    CHECK_INPUT(rboxes);
    return tools_rbox2rect_cuda(rboxes);
}

at::Tensor tools_rbox2corner(const at::Tensor &rboxes)
{
    /* rboxes: Tensor shape[N, 5]
     * */
    CHECK_INPUT(rboxes);
    return tools_rbox2corner_cuda(rboxes);
}

at::Tensor get_ngds_score(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const float &factor, const int &mode, const std::vector<int64_t> &block_size)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * factor: float
     * mode: int, Support mode: Normal(0), Shrink(1)
     * block_size: List[size_1, size_2, ...]
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(rboxes);
    return tool_normalize_gauss_distribution_score_cuda(xs, ys, rboxes, factor, mode, block_size);
}

at::Tensor get_ngds_score_v2(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const float &factor, const int &mode)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * factor: float
     * mode: int, Support mode: Normal(0), Shrink(1)
     * block_size: List[size_1, size_2, ...]
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(rboxes);
    return tool_normalize_gauss_distribution_score_cuda_v2(xs, ys, rboxes, factor, mode);
}

at::Tensor get_gds_score(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &rboxes, const float &factor, const int &mode, bool refined)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * factor: float
     * mode: int, Support mode: Normal(0), Shrink(1)
     * block_size: List[size_1, size_2, ...]
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(rboxes);
    return tool_gauss_distribution_score_cuda(xs, ys, rboxes, factor, mode, refined);
}

at::Tensor get_max_distance(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * factor: float
     * block_size: List[size_1, size_2, ...]
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(bboxes);
    return tool_max_distance_cuda(xs, ys, bboxes);
}

at::Tensor get_inside_regress_mask(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &bboxes, const at::Tensor &regress_ranges)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * factor: float
     * block_size: List[size_1, size_2, ...]
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(bboxes);
    CHECK_INPUT(regress_ranges);
    return tool_inside_regress_mask_cuda(xs, ys, bboxes, regress_ranges);
}

at::Tensor get_inside_balance_regress_mask(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &bboxes, const at::Tensor &rboxes,
        const at::Tensor &regress_ranges, const at::Tensor &strides,
        const float &factor)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * bboxes: Tensor shape[npoints, nrboxes, 4]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * regress_ranges: Tensor shape[npoints, nrboxes, 2]
     * strides: Tensor shape[npoints]
     * factor: float
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(bboxes);
    CHECK_INPUT(rboxes);
    CHECK_INPUT(regress_ranges);
    CHECK_INPUT(strides);
    return tool_inside_balance_regress_mask_cuda(xs, ys, bboxes, rboxes, regress_ranges, strides, factor);
}

at::Tensor get_inside_balance_regress_mask_v2(
        const at::Tensor &xs, const at::Tensor &ys,
        const at::Tensor &rboxes, const at::Tensor &regress_ranges,
        const at::Tensor &strides, const float &factor)
{
    /* xs: Tensor shape[npoints, nrboxes]
     * ys: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * regress_ranges: Tensor shape[npoints, nrboxes, 2]
     * strides: Tensor shape[npoints]
     * factor: float
     * */
    CHECK_INPUT(xs);
    CHECK_INPUT(ys);
    CHECK_INPUT(rboxes);
    CHECK_INPUT(regress_ranges);
    CHECK_INPUT(strides);
    return tool_inside_balance_regress_mask_cuda_v2(xs, ys, rboxes, regress_ranges, strides, factor);
}

//at::Tensor rebalance_weight(const at::Tensor &weight, const at::Tensor &label, const at::Tensor &idx, const std::vector<int64_t> &block_size)
//{
//    /* xs: Tensor shape[npoints, nrboxes]
//     * ys: Tensor shape[npoints, nrboxes]
//     * rboxes: Tensor shape[npoints, nrboxes, 5]
//     * factor: float
//     * angle_positive: bool
//     * block_size: List[size_1, size_2, ...]
//     * */
//    CHECK_INPUT(weight);
//    CHECK_INPUT(label);
//    CHECK_INPUT(idx);
//    return rebalance_weight_cuda(weight, label, idx, block_size);
//}

at::Tensor tools_get_inside_mask_with_gds(const at::Tensor &x, const at::Tensor &y,
                                          const at::Tensor &rboxes, const float nds_threshold,
                                          const at::Tensor &nds)
{
    /* x: Tensor shape[npoints, nrboxes]
     * y: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * nds_threshold: float
     * nds: Tensor shape[npoints, nrboxes]
     * */
    CHECK_INPUT(rboxes);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(nds);
    return tools_get_inside_mask_with_gds_cuda(x, y, rboxes, nds_threshold, nds);
}

std::vector<at::Tensor> tools_get_inside_mask_with_obj_gds(const at::Tensor &x, const at::Tensor &y,
                                                           const at::Tensor &rboxes, const float nds_threshold,
                                                           const at::Tensor &nds)
{
    /* x: Tensor shape[npoints, nrboxes]
     * y: Tensor shape[npoints, nrboxes]
     * rboxes: Tensor shape[npoints, nrboxes, 5]
     * nds_threshold: float
     * nds: Tensor shape[npoints, nrboxes]
     * */
    CHECK_INPUT(rboxes);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(nds);
    return tools_get_inside_mask_with_obj_gds_cuda(x, y, rboxes, nds_threshold, nds);
}

std::vector<at::Tensor> tools_get_inside_mask_with_obj(const at::Tensor &x, const at::Tensor &y, const at::Tensor &rboxes)
{
    /* rboxes: Tensor shape[N, 5]
     * */
    CHECK_INPUT(rboxes);
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    return tools_get_inside_mask_with_obj_cuda(x, y, rboxes);
}

at::Tensor get_expand_score(const at::Tensor &bboxes_ids, const at::Tensor &score, const int num_gt, const float &filled_value)
{
    /* bboxes: Tensor shape[npoints,]
     * num_gt: int
     * */
    CHECK_INPUT(bboxes_ids);
    CHECK_INPUT(score);
    return tools_get_expand_score_cuda(bboxes_ids, score, num_gt, filled_value);
}

at::Tensor get_keep_sample_idx(const at::Tensor &dynanic_k, const at::Tensor &topk_ids, const int &n_points)
{
    /* bboxes: Tensor shape[npoints,]
     * num_gt: int
     * */
    CHECK_INPUT(dynanic_k);
    CHECK_INPUT(topk_ids);
    return tools_get_keep_sample_idx_cuda(dynanic_k, topk_ids, n_points);
}

at::Tensor poly_cut(const at::Tensor &polys, const std::vector<int64_t> &image_size)
{
    CHECK_INPUT(polys);
    return tools_poly_cut_cuda(polys, image_size);
}

at::Tensor poly_cut_v2(const at::Tensor &polys, const std::vector<int64_t> &image_size, const float half_iou_thre)
{
    CHECK_INPUT(polys);
    return tools_poly_cut_v2_cuda(polys, image_size, half_iou_thre);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("compute_poly_iou", &tools_compute_poly_iou, "tools_compute_poly_iou");
m.def("compute_rbox_iou", &tools_compute_rbox_iou, "tools_compute_poly_iou");
m.def("get_inside_mask", &tools_get_inside_mask, "get_inside_mask");
m.def("get_inside_mask_with_gds", &tools_get_inside_mask_with_gds, "get_inside_mask_with_gds");
m.def("get_inside_mask_with_obj", &tools_get_inside_mask_with_obj, "get_inside_mask_with_obj");
m.def("get_inside_mask_with_obj_gds", &tools_get_inside_mask_with_obj_gds, "get_inside_mask_with_obj_gds");
m.def("rbox2rect", &tools_rbox2rect, "rbox2rect");
m.def("rbox2corner", &tools_rbox2corner, "rbox2corner");
m.def("get_ngds_score", &get_ngds_score, "get_ngds_score");
m.def("get_ngds_score_v2", &get_ngds_score_v2, "get_ngds_score");
m.def("get_gds_score", &get_gds_score, "get_gds_score");
m.def("get_max_distance", &get_max_distance, "get_max_distance");
m.def("get_inside_regress_mask", &get_inside_regress_mask, "get_inside_regress_mask");
m.def("get_inside_balance_regress_mask", &get_inside_balance_regress_mask, "get_inside_balance_regress_mask");
m.def("get_inside_balance_regress_mask_v2", &get_inside_balance_regress_mask_v2, "get_inside_balance_regress_mask");
m.def("expand_score", get_expand_score, "expand_bbox_ids");
m.def("get_keep_sample_idx", get_keep_sample_idx, "get_keep_sample_idx");
m.def("poly_cut", poly_cut, "poly_cut");
m.def("poly_cut_v2", poly_cut_v2, "poly_cut_v2");
}