import numpy as np
import cv2
import os
from shapely.geometry import Polygon
import time

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from DetectionResult import DetectionResult
from gtholebox import GTHoleBox
from typing import List, Dict


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
    src_trans_np = np.asarray(src.points, dtype=np.uint)

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
    To convert the data type to the one for visualization purpose using opencv_plot2 function
    '''
    list_of_bbox = []
    idx = 0
    for _ in range(int((in_array.shape[0])/2)):
        list_of_bbox.append([in_array[idx][0], in_array[idx][1], in_array[idx+1][0], in_array[idx+1][1]])
        idx = idx + 2

    return list_of_bbox


def opencv_plot(img_, list_of_bbox_det, list_of_bbox_gt, win_name='default',
                save_path=None):
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

def opencv_plot2(img_, list_of_bbox_det, list_of_bbox_gt,  win_name='default',
                 save_path=None, show=True):

    idx = 0 
    for points_trg in list_of_bbox_gt:
        # print('Points :', points_trg)
        cv2.rectangle(img_, (int(points_trg[0]), int(points_trg[1])), (int(points_trg[2]), int(points_trg[3])), (0,0,255), 2)
        cv2.putText(img_, str(idx), (int(points_trg[0]), int(points_trg[1])), 1,1,  (0,0,255), 1)
        idx += 1

    idx = 0 
    for idx, points_src in enumerate(list_of_bbox_det):        
        cv2.rectangle(img_, (int(points_src[0]), int(points_src[1])), (int(points_src[2]), int(points_src[3])), (255,0,0), 2)
        cv2.putText(img_, str(idx), (int(points_src[0]), int(points_src[1])), 1,1, (255,0,0), 1)
        idx += 1
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


# --- NEW OPTIMIZED FUNCTIONS INJECTED HERE ---

def get_optimal_translation_voting(source, target, tolerance=5.0):
    """
    Finds the optimal 2D translation between two point sets by finding the 
    most common displacement vector, ignoring outliers and missing points.
    """
    if len(source) == 0 or len(target) == 0:
        return np.array([0.0, 0.0])

    # Compute all pairwise differences using broadcasting
    diffs = target[np.newaxis, :, :] - source[:, np.newaxis, :]
    diffs = diffs.reshape(-1, 2)
    
    # Round to group similar translations together
    rounded_diffs = np.round(diffs / tolerance) * tolerance
    
    # Find the most frequent translation vector
    unique_vectors, counts = np.unique(rounded_diffs, axis=0, return_counts=True)
    best_translation_binned = unique_vectors[np.argmax(counts)]
    
    # Refine to the exact mean of the raw vectors in the winning bin
    mask = np.all(rounded_diffs == best_translation_binned, axis=1)
    exact_translation = np.mean(diffs[mask], axis=0)
    
    return exact_translation

def icp2_zahid(source, target, img=None):
    """
    Perform alignment using Vector Voting and single-pass Hungarian Algorithm.
    (img parameter added to prevent crashes when called with img.copy() from transform_gt_bbox)
    """
    match_thresh = 50

    # Handle edge case of empty arrays
    if len(source) == 0 or len(target) == 0:
        return np.array([0.0, 0.0]), 1000000, source, target

    # start = time.perf_counter()

    # 1. Find the global shift using vector voting
    T = get_optimal_translation_voting(source, target, tolerance=50.0)
    # T = get_optimal_translation_voting(source, target, tolerance=50.0)
    
    # print(f"\n[VOTING] Time: {(time.perf_counter()-start)*1000:.3f} ms")
    
    # 2. Apply the shift to the ground truth
    transformed_source = source + T
    
    # 3. Run Hungarian matching EXACTLY ONCE on the shifted points
    dist_matrix = distance.cdist(transformed_source, target, 'euclidean')
    
    # Penalize distances beyond threshold to prevent forced bad matches
    dist_matrix[dist_matrix > match_thresh] = 10000 
    
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Filter out the forced matches that exceed our threshold
    valid_mask = dist_matrix[row_ind, col_ind] < match_thresh
    src_idx = row_ind[valid_mask]
    tgt_idx = col_ind[valid_mask]
    
    if len(src_idx) == 0:
        # Fallback if no valid matches are found
        return T, 1000000, transformed_source, target

    final_error = dist_matrix[src_idx, tgt_idx].sum()
    
    final_transformed_src = transformed_source[src_idx]
    final_matched_tgt = target[tgt_idx]
    
    return T, final_error, final_transformed_src, final_matched_tgt

# --- END OPTIMIZED FUNCTIONS ---


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
    
    # print(f"Length Keys in gt bbox dict : {list(gt_bbox_dict.keys())}")
    
    for cam_id in range(total_cams-1):
        ret_hole_id_dict[cam_id] = []
        
        # Skip if camera ID not in dictionaries
        if cam_id not in gt_bbox_dict or cam_id not in detection_dict:
            print(f"Camera ID {cam_id} not found in gt_bbox_dict or detection_dict")
            ret_hole_id_dict[cam_id] = hole_id_dict.get(cam_id, [])
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
                    T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target)
                    print(f"\n[VOTING] Time: {(time.perf_counter()-start)*1000:.3f} ms")
                    # T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target, img.copy())
                else:
                    start = time.perf_counter()

                    T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target)
                    print(f"\n[VOTING] Time: {(time.perf_counter()-start)*1000:.3f} ms")


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
                            
                
                if DISPLAY or SAVE_RES:
                    # Original image with GT boxes
                    original_img = opencv_plot2(
                        gt_img.copy(),
                        [],  # no detections
                        one_array_to_list_of_bbox2(gt_bbox_2d),
                        show=False
                    )

                    # RIGHT: Target image + detections + ICP result
                    icp_img = opencv_plot2(
                        target_img.copy(),
                        one_array_to_list_of_bbox2(det_bbox_2d),
                        one_array_to_list_of_bbox2(transformed_gt_bbox),
                        show=False
                    )

                    # Add labels
                    cv2.putText(original_img, "Ground Truth", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(icp_img, "Detections + ICP Result", (20, 40),
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
                if (w > 0) and (w < 2471) and (h > 0) and (h < 2063):
                    if (bb[0] < 10000) and (bb[1] < 10000) and (bb[2] < 10000) and (bb[3] < 10000) :

                        '''
                        [max(int(bbox[1]) - additional_patch, 0):
                        min(int(bbox[3] + additional_patch), 2063),
                        max(int(bbox[0]) - additional_patch, 0):
                        min(int(bbox[2]) + additional_patch, 2473)]
                        if (image_patch.shape[0] > 0) and (image_patch.shape[1] > 0):
                        '''
                        y_min = max(int(bb[1]) - additional_patch, 0)
                        y_max = min(int(bb[3] + additional_patch), 2063)
                        x_min = max(int(bb[0]) - additional_patch, 0)
                        x_max = min(int(bb[2]) + additional_patch, 2473)

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

    return transformed_gt_array, ret_hole_id_dict


# def transform_gt_bbox_homography_zahid2(
#     total_cams : int,
#     gt_bbox_dict: Dict[int, List[GTHoleBox]],    # gt holes by cam id, gt holes are List[GTHoleBox]
#     detection_dict: Dict[int, List[DetectionResult]],  # detection_dict by camid, List[DetectionResult]
#     hole_id_dict,    # gt hole IDs by cam id, hole IDs are List[int]
#     img_dict={},
#     outdir='icp_out/',
#     SAVE_RES=False,
#     DISPLAY=False
# ):
#     additional_patch = 10
#     transformed_gt_array = []
#     ret_hole_id_dict = {}
    
#     # print(f"Length Keys in gt bbox dict : {list(gt_bbox_dict.keys())}")
    
#     for cam_id in range(total_cams-1):
#         ret_hole_id_dict[cam_id] = []
        
#         # Skip if camera ID not in dictionaries
#         if cam_id not in gt_bbox_dict or cam_id not in detection_dict:
#             print(f"Camera ID {cam_id} not found in gt_bbox_dict or detection_dict")
#             ret_hole_id_dict[cam_id] = hole_id_dict.get(cam_id, [])
#             transformed_gt_array.append([])
#             continue
            
#         det_box_arr = [detect.box for detect in detection_dict[cam_id]]
#         gt_box_arr = [gt_box.get_hole_box() for gt_box in gt_bbox_dict[cam_id]]
#         det_bbox_2d, det_centers_2d = convert_bbox_to_ndarray(det_box_arr)
#         gt_bbox_2d, gt_centers_2d = convert_bbox_to_ndarray(gt_box_arr)
#         if DISPLAY:
#             img = img_dict[cam_id]

#         target = det_centers_2d
#         source = gt_centers_2d

#         transformed_gt_bbox = gt_bbox_2d.copy()

#         if det_bbox_2d.shape[0] > 0:
#             try:
#                 if DISPLAY:
#                     opencv_plot2(img.copy(), one_array_to_list_of_bbox2(det_bbox_2d), one_array_to_list_of_bbox2(gt_bbox_2d), win_name='Initial Alignment')
#                     T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target, img.copy())
#                 else:
#                     T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target)

#                 transformed_gt_bbox = gt_bbox_2d + T

#                 temp_source = np.float32(trans_src)
#                 temp_target = np.float32(trans_tgt)

#                 if temp_target.shape[0] > 3:
#                     T, _ = cv2.estimateAffine2D(temp_source, temp_target, False)
                    
#                     if T is not None:
#                         affine_fit_error = np.sum(np.linalg.norm((cv2.transform(np.array([temp_source]), T)[0]) - temp_target, axis=1))

#                         if affine_fit_error + 50 < baseline_fit_error:
#                             transformed_gt_bbox_affine = cv2.transform(np.array([transformed_gt_bbox]), T)[0]
#                             transformed_gt_bbox = check_bbox_flip(transformed_gt_bbox_affine, transformed_gt_bbox)
                            
#                 if DISPLAY:
#                     result_img = opencv_plot2(img.copy(), one_array_to_list_of_bbox2(det_bbox_2d), one_array_to_list_of_bbox2(transformed_gt_bbox), win_name='P2P ICP')

#                     if SAVE_RES:
#                         cv2.imwrite(f'{outdir}/{cam_id}.jpg', result_img)

#                 if DISPLAY:
#                     cv2.waitKey(1)
#             except Exception as e:
#                 print(f"Error in ICP processing for camera {cam_id}: {e}")
#                 # Use original GT bbox if ICP fails
#                 transformed_gt_bbox = gt_bbox_2d.copy()

#         transf_array = one_array_to_list_of_bbox(transformed_gt_bbox)
#         confirmed_array = []
#         hole_id_array = hole_id_dict.get(cam_id, [])

#         for bi, bb in enumerate(transf_array):
#             bb = cvt_bbox(bb)
#             if len(bb) > 0:
#                 w = int(bb[2])-int(bb[0])
#                 h = int(bb[3])-int(bb[1])
#                 if (w > 0) and (w < 2471) and (h > 0) and (h < 2063):
#                     if (bb[0] < 10000) and (bb[1] < 10000) and (bb[2] < 10000) and (bb[3] < 10000) :

#                         y_min = max(int(bb[1]) - additional_patch, 0)
#                         y_max = min(int(bb[3] + additional_patch), 2063)
#                         x_min = max(int(bb[0]) - additional_patch, 0)
#                         x_max = min(int(bb[2]) + additional_patch, 2473)

#                         if ((x_max-x_min) > 0) and (y_max-y_min) > 0:
#                             confirmed_array.append(bb)
#                             if bi < len(hole_id_array):
#                                 ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                        
#                         else:# Send GT HOLE
#                             print(f"ICP sent the hole outside the image boundaries ((x_max-x_min) > 0) and (y_max-y_min) > 0")
#                             if bi < len(one_array_to_list_of_bbox(gt_bbox_2d)):
#                                 confirmed_array.append(one_array_to_list_of_bbox(gt_bbox_2d)[bi])
#                                 if bi < len(hole_id_array):
#                                     ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                            
#                     else: # Send GT HOLE
#                         print(f"ICP sent the hole outside the image boundaries")
                        
#                         if bi < len(one_array_to_list_of_bbox(gt_bbox_2d)):
#                             confirmed_array.append(one_array_to_list_of_bbox(gt_bbox_2d)[bi])
#                             if bi < len(hole_id_array):
#                                 ret_hole_id_dict[cam_id].append(hole_id_array[bi])
                    
#         transformed_gt_array.append(confirmed_array)

#     return transformed_gt_array, ret_hole_id_dict


def image_patch_for_cls_transformed(images, transformed_gt_bbox_array):
    additional_patch = 10
    image_patches = {}
    for cam_i, bbox_array in enumerate(transformed_gt_bbox_array):
        image_patches[cam_i] = []

        for bi, bbox in enumerate(bbox_array):
            image_patch = images[cam_i][max(int(bbox[1])-additional_patch, 0):
                                        min(int(bbox[3]+additional_patch), 2063),
                          max(int(bbox[0])-additional_patch, 0):
                          min(int(bbox[2])+additional_patch, 2473)]
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