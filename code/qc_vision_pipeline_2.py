import numpy as np
import cv2
import os 
import time
from shapely.geometry import Polygon

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance, cKDTree

from DetectionResult import DetectionResult
from gtholebox import GTHoleBox
from typing import List, Dict

# Panel image dimensions (kept as named constants so the bounds check and the
# patch-clipping logic can't silently drift apart, as they had before: one used
# 2471x2063 for the bounds check and another spot used 2473x2063 for clipping).
IMG_WIDTH = 2473
IMG_HEIGHT = 2063


def convert_bbox_to_ndarray(list_of_bbox: List[float]):
    out_centers = np.zeros((len(list_of_bbox),2))
    out_array = np.zeros((len(list_of_bbox)*2,2))

    for idx, rect in enumerate(list_of_bbox):        
        out_centers[idx, :] = (rect[0] + rect[2]) / 2 , (rect[1] + rect[3]) / 2
        out_array[idx*2,:] = rect[0], rect[1]        
        out_array[idx*2+1,:] = rect[2], rect[3]

    return out_array, out_centers


def transform_icp(src, transformation):
    src.transform(transformation)
    # NOTE: previously cast to np.uint, which silently wraps any negative
    # coordinate (very possible after a real transform) into a huge positive
    # number instead of raising an error. float preserves the true values;
    # cast to int only at the point where you need pixel indices.
    src_trans_np = np.asarray(src.points, dtype=np.float64)

    return src_trans_np


def one_array_to_list_of_bbox(in_array):
    list_of_bbox = []
    idx = 0
    for _ in range(int(len(in_array) / 2)):
        list_of_bbox.append([in_array[idx][0], in_array[idx][1], in_array[idx + 1][0], in_array[idx + 1][1]])
        idx = idx + 2

    return list_of_bbox


def one_array_to_list_of_bbox2(in_array):
    '''
    Was a byte-for-byte duplicate of one_array_to_list_of_bbox (only used
    .shape[0] instead of len(), which is equivalent for a numpy array).
    Kept as a thin alias so existing call sites don't need to change.
    '''
    return one_array_to_list_of_bbox(in_array)


def opencv_plot(img_, list_of_bbox_det, list_of_bbox_gt, win_name='default'):
    for points_trg in list_of_bbox_gt:
        print('Points :', points_trg)
        cv2.rectangle(img_, (int(points_trg[0]), int(points_trg[1])), (int(points_trg[2]), int(points_trg[3])),
                      (0, 0, 255), 2)

    for idx, points_src in enumerate(list_of_bbox_det):
        cv2.rectangle(img_, (int(points_src[0]), int(points_src[1])), (int(points_src[2]), int(points_src[3])),
                      (255, 0, 0), 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img_)
    cv2.waitKey(1)

# def opencv_plot2(img_, list_of_bbox_det, list_of_bbox_gt,  win_name='default',
#                  show=True, hole_ids=None):

#     idx = 0 
#     for points_trg, hole_id in zip(list_of_bbox_gt, hole_ids):
#         # print('Points :', points_trg)
#         cv2.rectangle(img_, (int(points_trg[0]), int(points_trg[1])), (int(points_trg[2]), int(points_trg[3])), (0,0,255), 2)
#         cv2.putText(img_, str(hole_id), (int(points_trg[0]), int(points_trg[1])), 1,1,  (0,0,255), 1)
#         idx += 1

#     idx = 0 
#     for idx, points_src in enumerate(list_of_bbox_det):        
#         cv2.rectangle(img_, (int(points_src[0]), int(points_src[1])), (int(points_src[2]), int(points_src[3])), (255,0,0), 2)
#         cv2.putText(img_, str(idx), (int(points_src[0]), int(points_src[1])), 1,1, (255,0,0), 1)
#         idx += 1
#     if show:
#         cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#         cv2.imshow(win_name, img_)
#         cv2.waitKey(1)

#     return img_

def opencv_plot2(
    img_,
    list_of_bbox_det,
    list_of_bbox_gt,
    win_name='default',
    show=True,
    hole_ids=None,
    det_labels=None
):

    if hole_ids is None:
        hole_ids = [""] * len(list_of_bbox_gt)

    if det_labels is None:
        det_labels = [str(i) for i in range(len(list_of_bbox_det))]

    # Draw ICP (GT) boxes
    for points_trg, hole_id in zip(list_of_bbox_gt, hole_ids):

        cv2.rectangle(
            img_,
            (int(points_trg[0]), int(points_trg[1])),
            (int(points_trg[2]), int(points_trg[3])),
            (0, 0, 255),
            2,
        )

        if hole_id != "":
            cv2.putText(
                img_,
                str(hole_id),
                (int(points_trg[0]), int(points_trg[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    # Draw detections
    for idx, points_src in enumerate(list_of_bbox_det):

        cv2.rectangle(
            img_,
            (int(points_src[0]), int(points_src[1])),
            (int(points_src[2]), int(points_src[3])),
            (255, 0, 0),
            2,
        )

        label = det_labels[idx] if idx < len(det_labels) else str(idx)
        # cv2.putText(
        #     img_,
        #     str(label),
        #     (int(points_src[0]), int(points_src[1])),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 0, 0),
        #     2,
        # )

    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img_)
        cv2.waitKey(1)

    return img_

def closest_point_matching2(source, target, match_thesh = 50):
    hung_mat = distance.cdist(source, target, 'euclidean') # is the much faster version of above loop based version

    hung_mat[np.where(hung_mat > match_thesh)] = 1000 # to debug the issue related to linear_sum_assignment returning wrong  values

    row_ind, col_ind = linear_sum_assignment(hung_mat, maximize=False) 
    
    #replacing above with vectorization
    valid_mask = hung_mat[row_ind, col_ind] < match_thesh
    row_ind_new = row_ind[valid_mask]
    col_ind_new = col_ind[valid_mask]

    return row_ind_new, col_ind_new, hung_mat[row_ind, col_ind].sum() #row_ind, col_ind


# --- UPGRADED HIGH-PRECISION ALIGNMENT FUNCTIONS ---

def get_optimal_translation_voting(source, target, tolerance=5.0):
    """
    Finds the optimal 2D translation using continuous density estimation (KD-Tree)
    to guarantee finding the true mode without histogram bin-splitting artifacts.
    """
    if len(source) == 0 or len(target) == 0:
        return np.array([0.0, 0.0])

    # Compute all pairwise difference vectors
    diffs = target[np.newaxis, :, :] - source[:, np.newaxis, :]
    diffs = diffs.reshape(-1, 2)
    
    # Use KD-Tree to find the densest cluster in Euclidean space
    tree = cKDTree(diffs)
    
    # Count how many displacement vectors fall within the tolerance radius for EACH vector
    counts = tree.query_ball_point(diffs, r=tolerance, return_length=True)
    
    # The true translation is the vector with the highest consensus
    best_idx = np.argmax(counts)
    best_diff = diffs[best_idx]
    
    # Refine by taking the exact mean of all vectors belonging to this cluster
    neighbors = tree.query_ball_point(best_diff, r=tolerance)
    exact_translation = np.mean(diffs[neighbors], axis=0)
    
    return exact_translation

def icp2_zahid(source, target, img=None):
    """
    Perform high-accuracy alignment using Density Voting and a 2-Pass Refinement.
    """
    match_thresh = 50
    empty_idx = np.array([], dtype=int)

    if len(source) == 0 or len(target) == 0:
        return np.array([0.0, 0.0]), 1000000, source, target, empty_idx, empty_idx

    # --- PASS 1: Global Shift via Continuous Density Voting ---
    T = get_optimal_translation_voting(source, target, tolerance=50.0) 
    transformed_source = source + T
    
    dist_matrix = distance.cdist(transformed_source, target, 'euclidean')
    dist_matrix[dist_matrix > match_thresh] = 10000 
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    valid_mask = dist_matrix[row_ind, col_ind] < match_thresh
    src_idx = row_ind[valid_mask]
    tgt_idx = col_ind[valid_mask]
    
    if len(src_idx) == 0:
        return T, 1000000, transformed_source, target, empty_idx, empty_idx

    # --- PASS 2: Inlier Refinement ---
    # Calculate the exact translation ONLY between the confirmed matched pairs.
    # This acts as a 1-step ICP that eliminates edge-skew caused by micro-rotations.
    refined_diffs = target[tgt_idx] - source[src_idx]
    refined_T = np.mean(refined_diffs, axis=0)
    
    # Apply refined shift and perform the final strict Hungarian match
    final_transformed_source = source + refined_T
    
    final_dist_matrix = distance.cdist(final_transformed_source, target, 'euclidean')
    final_dist_matrix[final_dist_matrix > match_thresh] = 10000
    
    f_row_ind, f_col_ind = linear_sum_assignment(final_dist_matrix)
    f_valid_mask = final_dist_matrix[f_row_ind, f_col_ind] < match_thresh
    
    f_src_idx = f_row_ind[f_valid_mask]
    f_tgt_idx = f_col_ind[f_valid_mask]
    
    if len(f_src_idx) == 0:
        return refined_T, 1000000, final_transformed_source, target, empty_idx, empty_idx

    final_error = final_dist_matrix[f_src_idx, f_tgt_idx].sum()
    final_matched_src = final_transformed_source[f_src_idx]
    final_matched_tgt = target[f_tgt_idx]
    
    return (
        refined_T,
        final_error,
        final_matched_src,
        final_matched_tgt,
        f_src_idx,
        f_tgt_idx
    )

# --- END UPGRADED FUNCTIONS ---


def check_bbox_flip(new_bbox, old_bbox, th=4) :
    '''
    check if the updated bounding boxes are flipped (cv2.affineTransform does that sometimes)

    Args:
    - new_bbox : bbounding box which is updated by affine transform
    - old_bbox : bbounding box before doing the afffine transform
    '''
    idx = 0
    # for _ in range(int(new_bbox.shape[0]/2)): # just checking one bbox is enough, because all the bboxes flips with same magnitude
    if abs(new_bbox[idx][0] - new_bbox[idx+1][0]) < th or abs(new_bbox[idx][1] - new_bbox[idx+1][1]) < th:            
        return old_bbox
    elif abs((new_bbox[idx+1][0]-new_bbox[idx][0]) / (new_bbox[idx+1][1]-new_bbox[idx][1])) > 20 or abs((new_bbox[idx+1][0]-new_bbox[idx][0]) / (new_bbox[idx+1][1]-new_bbox[idx][1])) < 1/20:
        return old_bbox
        
    return new_bbox

def cvt_bbox(bbox):
    minx = min(bbox[0], bbox[2])
    miny = min(bbox[1], bbox[3])

    maxx = max(bbox[0], bbox[2])
    maxy = max(bbox[1], bbox[3])
    return [minx, miny, maxx, maxy]


def transform_gt_bbox_homography_zahid2(
    total_cams : int,
    gt_bbox_dict: Dict[int, List[GTHoleBox]],    # gt holes by cam id, gt holes are List[GTHoleBox]
    detection_dict: Dict[int, List[DetectionResult]],  # detection_dict by camid, List[DetectionResult]
    hole_id_dict,    # gt hole IDs by cam id, hole IDs are List[int]
    img_dict={},
    outdir='icp_out/',
    SAVE_RES=False,
    DISPLAY=False,
    output_filename = None
):
    
    # print("DISPLAY =", DISPLAY)
    # print("SAVE_RES =", SAVE_RES)


    additional_patch = 10
    transformed_gt_array = []
    ret_hole_id_dict = {}
    # Hole IDs whose GT box did NOT get a Hungarian-matched detection within
    # match_thresh. This is the "unmatched" list for the QC report — separate
    # from ret_hole_id_dict, which includes every in-bounds hole regardless of
    # whether it was actually matched.
    unmatched_hole_id_dict = {}
    
    # print(f"Length Keys in gt bbox dict : {list(gt_bbox_dict.keys())}")
    
    for cam_id in range(total_cams):
        ret_hole_id_dict[cam_id] = []
        
        # Skip if camera ID not in dictionaries
        if cam_id not in gt_bbox_dict or cam_id not in detection_dict:
            print(f"Camera ID {cam_id} not found in gt_bbox_dict or detection_dict")
            ret_hole_id_dict[cam_id] = hole_id_dict.get(cam_id, [])
            # No detections/GT to work with at all -> every hole counts as unmatched.
            unmatched_hole_id_dict[cam_id] = list(hole_id_dict.get(cam_id, []))
            transformed_gt_array.append([])
            continue
            
        det_box_arr = [detect.box for detect in detection_dict[cam_id]]
        gt_box_arr = [gt_box.get_hole_box() for gt_box in gt_bbox_dict[cam_id]]
        det_bbox_2d, det_centers_2d = convert_bbox_to_ndarray(det_box_arr)
        gt_bbox_2d, gt_centers_2d = convert_bbox_to_ndarray(gt_box_arr)
        if DISPLAY or SAVE_RES:
                gt_img = img_dict[cam_id]["gt"]
                target_img = img_dict[cam_id]["target"]

        target = det_centers_2d
        source = gt_centers_2d

        transformed_gt_bbox = gt_bbox_2d.copy()

        # Default to "nothing matched" so this is always defined, whether
        # there are zero detections, or an exception fires before/while ICP runs.
        matched_gt_idx = np.array([], dtype=int)
        matched_det_idx = np.array([], dtype=int)

        if det_bbox_2d.shape[0] > 0:
            try:
                if DISPLAY:
                    # opencv_plot2(img.copy(), 
                    #              one_array_to_list_of_bbox2(det_bbox_2d), 
                    #              one_array_to_list_of_bbox2(gt_bbox_2d), 
                    #              win_name='Initial Alignment')

                    # draw_registration_result(source, target, trans_init, img.copy(), len(xco), win_name='Initial Alignment')
                    # print('Source Shape : ', source.shape)
            # print('Target shape :', target.shape)
                    start = time.perf_counter()
                    T, baseline_fit_error, trans_src, trans_tgt, matched_gt_idx, matched_det_idx = icp2_zahid(source, target)
                    print(f"\n[New ICP] Time: {(time.perf_counter()-start)*1000:.3f} ms")
                    # T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target, img.copy())
                else:
                    start = time.perf_counter()

                    T, baseline_fit_error, trans_src, trans_tgt, matched_gt_idx, matched_det_idx = icp2_zahid(source, target)
                    print(f"\n[New ICP] Time: {(time.perf_counter()-start)*1000:.3f} ms")


                transformed_gt_bbox = gt_bbox_2d + T

                temp_source = np.float32(trans_src)
                temp_target = np.float32(trans_tgt)

                if temp_target.shape[0] > 3:
                    T, _ = cv2.estimateAffine2D(temp_source, temp_target, False)
                    
                    if T is not None:
                        # if method == 'AFFINE': # aftr evaluation on the raw images dataset, affine is the best finetuning method so far
                        affine_fit_error = np.sum(np.linalg.norm((cv2.transform(np.array([temp_source]), T)[0]) - temp_target, axis=1))

                        if affine_fit_error + 50 < baseline_fit_error:
                            transformed_gt_bbox_affine = cv2.transform(np.array([transformed_gt_bbox]), T)[0]
                            transformed_gt_bbox = check_bbox_flip(transformed_gt_bbox_affine, transformed_gt_bbox)
                            
                # ----------------------------------------------------------
                # Build hole IDs using Hungarian matching
                # ----------------------------------------------------------
                hole_id_array = hole_id_dict.get(cam_id, [])

                # One entry per transformed GT box
                hungarian_hole_ids = [""] * len(one_array_to_list_of_bbox2(transformed_gt_bbox))

                # Fill only matched GT boxes with their original hole IDs
                # Also build matching labels for the detection boxes so the
                # "Detections + NEW ICP" panel shows which hole each detection
                # was actually matched to (unmatched detections keep a "?idx" label).
                det_labels = [f"?{i}" for i in range(len(det_bbox_2d) // 2)]
                for gt_idx, det_idx in zip(matched_gt_idx, matched_det_idx):
                    if gt_idx < len(hole_id_array):
                        hungarian_hole_ids[gt_idx] = hole_id_array[gt_idx]
                        if det_idx < len(det_labels):
                            det_labels[det_idx] = str(hole_id_array[gt_idx])

                if DISPLAY or SAVE_RES:
                    # Original image with GT boxes
                    original_img = opencv_plot2(
                        gt_img.copy(),
                        [],  # no detections
                        one_array_to_list_of_bbox2(gt_bbox_2d),
                        show=False,
                        hole_ids=hole_id_dict.get(cam_id, [])
                    )

                    # RIGHT: Target image + detections + ICP result
                    icp_img = opencv_plot2(
                        target_img.copy(),
                        one_array_to_list_of_bbox2(det_bbox_2d),
                        one_array_to_list_of_bbox2(transformed_gt_bbox),
                        show=False,
                        hole_ids=hungarian_hole_ids,
                        det_labels=det_labels
                    )

                    # Add labels
                    cv2.putText(original_img, "Ground Truth", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(icp_img, "Detections + NEW ICP", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Stack horizontally
                    combined_img = np.hstack((original_img, icp_img))

                    cv2.namedWindow("ICP Comparison", cv2.WINDOW_NORMAL)
                    cv2.imshow("ICP Comparison", combined_img)
                    # print("ENTERED VISUALIZATION BLOCK")
                    if SAVE_RES:
                        os.makedirs(outdir, exist_ok=True)
                        save_path = os.path.join(
                            outdir,
                            output_filename if output_filename else f"{cam_id}.jpg"
                        )

                        ok = cv2.imwrite(save_path, combined_img)

                        print(
                            f"Save {'SUCCESS' if ok else 'FAILED'}: {save_path}"
                        )

                if DISPLAY:
                    cv2.waitKey(0)
            except Exception as e:
                print(f"Error in ICP processing for camera {cam_id}: {e}")
                # Use original GT bbox if ICP fails
                transformed_gt_bbox = gt_bbox_2d.copy()

        # return one_array_to_list_of_bbox(transformed_det_bbox)
        transf_array = one_array_to_list_of_bbox(transformed_gt_bbox)
        confirmed_array = []
        hole_id_array = hole_id_dict.get(cam_id, [])

        for bi, bb in enumerate(transf_array):
            #print("transform bb", bb)
            bb = cvt_bbox(bb)
            if len(bb) > 0:
                w = int(bb[2])-int(bb[0])
                h = int(bb[3])-int(bb[1])
                #print("transform bb", bb, w, h)
                if (w > 0) and (w < IMG_WIDTH) and (h > 0) and (h < IMG_HEIGHT):
                    if (bb[0] < 10000) and (bb[1] < 10000) and (bb[2] < 10000) and (bb[3] < 10000) :

                        '''
                        [max(int(bbox[1]) - additional_patch, 0):
                        min(int(bbox[3] + additional_patch), 2063),
                        max(int(bbox[0]) - additional_patch, 0):
                        min(int(bbox[2]) + additional_patch, 2473)]
                        if (image_patch.shape[0] > 0) and (image_patch.shape[1] > 0):
                        '''
                        y_min = max(int(bb[1]) - additional_patch, 0)
                        y_max = min(int(bb[3] + additional_patch), IMG_HEIGHT)
                        x_min = max(int(bb[0]) - additional_patch, 0)
                        x_max = min(int(bb[2]) + additional_patch, IMG_WIDTH)

                        if ((x_max-x_min) > 0) and (y_max-y_min) > 0:
                            #print("transfirm", bb, w, h)
                            confirmed_array.append(bb)
                            if bi < len(hole_id_array):
                                ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                        
                        else:# Send GT HOLE
                            print(f"ICP sent the hole outside the image boundaries ((x_max-x_min) > 0) and (y_max-y_min) > 0")
                            if bi < len(one_array_to_list_of_bbox(gt_bbox_2d)):
                                confirmed_array.append(one_array_to_list_of_bbox(gt_bbox_2d)[bi])
                                if bi < len(hole_id_array):
                                    ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                            
                            
                    else: # Send GT HOLE
                        print(f"ICP sent the hole outside the image boundaries")
                        
                        if bi < len(one_array_to_list_of_bbox(gt_bbox_2d)):
                            confirmed_array.append(one_array_to_list_of_bbox(gt_bbox_2d)[bi])
                            if bi < len(hole_id_array):
                                ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                    
        # ----------------------------------------------------------
        # Unmatched hole IDs for this camera: any GT hole whose index
        # never showed up in matched_gt_idx (i.e. Hungarian assignment
        # found no detection for it within match_thresh).
        # ----------------------------------------------------------
        matched_idx_set = set(int(i) for i in matched_gt_idx)
        unmatched_hole_id_dict[cam_id] = [
            hole_id_array[i]
            for i in range(len(hole_id_array))
            if i not in matched_idx_set
        ]

        #transformed_gt_array.append(one_array_to_list_of_bbox(transformed_det_bbox))
        transformed_gt_array.append(confirmed_array)

        '''
        f = open('./icp_check/trans_' + str(cam_id) + '.txt', 'w')
        for tb in confirmed_array:
            f.write(str(tb[0]) + ' ' + str(tb[1]) +' '+ str(tb[2]) +' '+ str(tb[3]) + '\n')
        f.close()
        '''

        #print("comfirmed array", confirmed_array)
        #for cam_i, bbox_array in enumerate(transformed_gt_bbox_array):
        #    image_patches[cam_i] = []
        #    for bi, bbox in enumerate(bbox_array):

    return transformed_gt_array, ret_hole_id_dict, unmatched_hole_id_dict


def image_patch_for_cls_transformed(images, transformed_gt_bbox_array):
    additional_patch = 10
    image_patches = {}
    for cam_i, bbox_array in enumerate(transformed_gt_bbox_array):
        image_patches[cam_i] = []

        for bi, bbox in enumerate(bbox_array):
            image_patch = images[cam_i][max(int(bbox[1])-additional_patch, 0):
                                        min(int(bbox[3]+additional_patch), IMG_HEIGHT),
                          max(int(bbox[0])-additional_patch, 0):
                          min(int(bbox[2])+additional_patch, IMG_WIDTH)]
            if (image_patch.shape[0] > 0)  and (image_patch.shape[1] > 0) :
                image_patches[cam_i].append(image_patch)
    return image_patches


def image_patch_for_cls(images, detection_dict, target_label):
    additional_patch = 10
    image_patches = {}
    for cam_i in detection_dict.keys():
        image_patches[cam_i] = []
        bboxes = detection_dict[cam_i]
        for bbox in bboxes:
            if bbox['label'] == target_label:
                [x1, y1, x2, y2] = bbox['bbox']
                image_patch = images[cam_i][int(y1) - additional_patch:int(y2 + additional_patch),
                              int(x1) - additional_patch:int(x2) + additional_patch]
                if (image_patch.shape[0] > 0) and (image_patch.shape[1] > 0):
                    image_patches[cam_i].append(image_patch)
    return image_patches