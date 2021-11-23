### Tools for FCOSR cuda version
#### install
```shell
python3 setup_tools.py install
```

#### Pytorch Api
**Module name: tools_cuda**

| Functions | Arguments | Description |
| - | - | - |
|compute_poly_iou|poly1, poly2|compute iou between 2 polys|
|compute_rbox_iou|rbox1, rbox2(opencv-python format)|compute iou between 2 rboxes|
|get_inside_mask|x, y, rboxes|determine if the point is in the rotation box|
|get_inside_mask_with_gds|x, y, rboxes, nds_threshold, nds|determine if the point is in the rotation box by nds|
|get_inside_mask_with_obj|x, y, rboxes|This function determine points in the rbox and return target id|
|get_inside_mask_with_obj_gds|x, y, rboxes, nds_threshold, nds|combine function get_inside_mask_with_gds and get_inside_mask_with_obj|
|rbox2rect|rbox|rbox to bbox, [xmin, ymin, xmax, ymax]|
|rbox2corner|rbox|rbox to corners(8-points)|
|get_ngds_score|x, y, rboxes, factor, mode, block_size|compute ngds score (force normalize)|
|get_ngds_score_v2|x, y, rboxes, factor, mode|get_ngds_score version 2 (unforced normalize)|
|get_gds_score|x, y, rboxes, factor, mode|compute gds score|
|get_max_distance|x, y, rbox|compute max distance|
|get_inside_regress_mask|x, y, bbox, regress_ranges|FCOS label assignment|
|get_inside_balance_regress_mask|x, y, bbox, rbox, regress_ranges, stride, factor|FCOSR label assignment|
|get_inside_balance_regress_mask_v2|x, y, rbox, regress_ranges, stride, factor|a simple version for FCOSR label assignment|
|expand_score|bboxes_ids, score, num_gt, filled_value|split score to each ids, component of simOTA and DropPS|
|get_keep_sample_idx|dynanic_k, topk_ids, n_points|component of simOTA and DropPS|
|poly_cut|polys, image_size|polygon cut, this function will cutoff part of ori-polygon which on the outside of image and create 4-ponits-polygon to instead it.|