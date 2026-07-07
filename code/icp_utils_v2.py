from tracemalloc import start
import time

import numpy as np
import cv2
from shapely.geometry import Polygon

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial import cKDTree

from DetectionResult import DetectionResult
from gtholebox import GTHoleBox
from typing import List, Dict
import os


os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
os.environ["XDG_SESSION_TYPE"] = "xcb"


# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["XDG_SESSION_TYPE"] = "wayland"
# os.environ["QT_QPA_PLATFORM"] = "wayland"
# os.environ["DISPLAY"] = ":0"     
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


import cv2

# Remove OpenCV Qt plugin path to avoid conflicts with PyQt5
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)


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
    # if save_path is not None:
    #     cv2.imwrite(save_path, img_)
    #     return
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
    
    # if save_path is not None:
    #     cv2.imwrite(save_path, img_)
    #     return
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, img_)
        cv2.waitKey(1)

    return img_

def closest_point_matching_mutual(
    source,
    target,
    match_thresh=30
):
    src_tree = cKDTree(source)
    # print(f"Source Tree : {src_tree}")
    tgt_tree = cKDTree(target)

    src_dist, src_nn = tgt_tree.query(source, k=1)
    tgt_dist, tgt_nn = src_tree.query(target, k=1)

    src_idx = []
    tgt_idx = []

    for s_idx, t_idx in enumerate(src_nn):

        if src_dist[s_idx] > match_thresh:
            continue

        if tgt_nn[t_idx] == s_idx:
            src_idx.append(s_idx)
            tgt_idx.append(t_idx)

    src_idx = np.array(src_idx)
    tgt_idx = np.array(tgt_idx)

    error = np.sum(
        np.linalg.norm(
            source[src_idx] - target[tgt_idx],
            axis=1
        )
    )

    return src_idx, tgt_idx, error

def closest_point_matching2(source, target, match_thesh = 30):  # need to optimize this func
    hung_mat = distance.cdist(source, target, 'euclidean') # is the much faster version of above loop based version
    # print(f'Hungarian Matrix : {hung_mat}')
    hung_mat[np.where(hung_mat > match_thesh)] = 10000 # to debug the issue related to linear_sum_assignment returning wrong  values
    # print(f'Hungarian Matrix after thresholding : {hung_mat}')
    row_ind, col_ind = linear_sum_assignment(hung_mat, maximize=False) 
    
    # row_ind_new = []
    # col_ind_new = []
    # for i in range(len(row_ind)):
    #     if hung_mat[row_ind[i]][col_ind[i]] < match_thesh :
    #         row_ind_new.append(row_ind[i])
    #         col_ind_new.append(col_ind[i])
    
    #replacing above with vectorization
    valid_mask = hung_mat[row_ind, col_ind] < match_thesh
    row_ind_new = row_ind[valid_mask]
    col_ind_new = col_ind[valid_mask]

    # print(f' Row : {row_ind} Cols : {col_ind}')

    # return len(row_ind_new), hung_mat[row_ind, col_ind].sum() #row_ind, col_ind
    return row_ind_new, col_ind_new, hung_mat[row_ind, col_ind].sum() #row_ind, col_ind

def icp2_zahid(source, target):
    """
    Perform the Iterative Closest Point (ICP) algorithm to align the source (GT) holes with the target (detected) holes.
    """
    prev_error = 1000000

    # final_transformed = source
    tranform = [0,0]
    closest_point_matching_threshold = 50

    # ordered_list = []
    ordered_set = set()
    break_hit = False
    final_trasnformed_src = []
    final_matched_tgt = []

    # Initialize cumulative transformation (identity matrix and zero translation)
    # print('Target pts :', target)
    # print('Src pts : ', source)

    for t_i, tgt_pts in enumerate(target): 

        for s_i, src_pts in enumerate(source):  

            # if f'{t_i}{s_i}' in ordered_list or f'{s_i}{t_i}' in ordered_list:
            #     continue
            key1 = f'{t_i},{s_i}'
            key2 = f'{s_i},{t_i}'
            if key1 in ordered_set or key2 in ordered_set:
                continue

            sub_dist = tgt_pts-src_pts

            # print('TGT PTS :', sub_dist)     

            transformed_source = source + sub_dist

            # src_idx, tgt_idx = closest_point_matching2(transformed_source, target)

            src_idx, tgt_idx, distances = closest_point_matching_mutual(transformed_source, target)
            # src_idx, tgt_idx, distances = closest_point_matching2(transformed_source, target)

            distances = abs(distances) + abs(len(src_idx)-len(source))*(closest_point_matching_threshold+5) # panelity factor in case all the holes are not matched

            # print('Indeices :', tgt_idx)
            matched_target = target[tgt_idx]
            transformed_source_ = transformed_source[src_idx]

            # distances = np.sum(np.linalg.norm(transformed_source_ - matched_target, axis=1))

            # print('Distances : ', distances)

            # ordered_list.append(f'{t_i}{s_i}')        
            ordered_set.add(key1)    

            if distances < prev_error:
                prev_error = distances                
                tranform = sub_dist
                final_trasnformed_src = transformed_source_
                final_matched_tgt = matched_target

                # opencv_plot3(img.copy(), matched_target, tgt_idx, transformed_source_, src_idx, distances,  win_name='interme')            

            if distances <= len(target)*1:
                # print('Break Hit !! at Distance :', distances)
                break_hit = True
                break

        if break_hit:
            break
        # print('Prev Error :', prev_error, 'Tranform :', tranform)
        # print('Source :', source)
        # print('Final Transformed Source :', final_trasnformed_src)
        # print('Final Matched Target :', final_matched_tgt)
    return tranform, prev_error, final_trasnformed_src, final_matched_tgt


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
                    print(f"\n[HUNGARIAN] Time: {(time.perf_counter()-start)*1000:.3f} ms")
                    # T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target, img.copy())
                else:
                    start = time.perf_counter()

                    T, baseline_fit_error, trans_src, trans_tgt = icp2_zahid(source, target)
                    print(f"\n[HUNGARIAN] Time: {(time.perf_counter()-start)*1000:.3f} ms")


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
