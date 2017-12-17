import ImageProcessing.process as proc
import cv2
import numpy as np
import pickle

dist_mat = pickle.load(open("ImageProcessing/dist_pickle.p", "rb" ))
mtx = dist_mat['mtx']
dist = dist_mat['dist']

def pipeline(img):
    l_found = r_found = False
    img = proc.undist(img, mtx, dist)
    
    for i in range(2):
        if i == 0:
            # default: fine lines from original image
            combined = proc.thresholding(img, {}, {})[0]
        else:
            # find lines from histogram equalized image
            combined = proc.thresholding(np.dstack([cv2.equalizeHist(img[:,:,0]), 
                                                cv2.equalizeHist(img[:,:,1]),
                                                cv2.equalizeHist(img[:,:,2])]), {}, {'b_chn_thresh':(135, 255)})[0]
        try:
            warped = cv2.warpPerspective(combined, pipeline.M, pipeline.img_size, flags=cv2.INTER_LINEAR)
            
        except AttributeError:
            h, w = img.shape[0], img.shape[1]
            pipeline.img_size = (w, h)
            
            src_pts = np.array([[1.7 * w // 10, h], [8.7 * w // 10, h], [5.75 * w // 10, h - h // 3],
                                [4.33 * w // 10, h - h // 3]])
            dst_pts = np.array([[2 * w // 10, h], [8 * w // 10, h], [8 * w // 10, h // 2], 
                                [2 * w // 10, h // 2]])
            
            warped, M, Minv = proc.perspective_transform(combined, src_pts, dst_pts)
            pipeline.M = M
            pipeline.Minv = Minv
            
            pipeline.left_lane = proc.Line(sample_window=5, margin=100)
            pipeline.right_lane = proc.Line(sample_window=5, margin=100)
                
        if pipeline.left_lane.detected and pipeline.right_lane.detected and \
            len(pipeline.left_lane.recent_x_fitted) != 0 and len(pipeline.right_lane.recent_x_fitted) != 0:
            warped[(pipeline.left_lane.mask(img) != 1) & (pipeline.right_lane.mask(img) != 1)] = 0            
            
        if pipeline.left_lane.detected and pipeline.right_lane.detected:
            y_pts = np.linspace(0, pipeline.img_size[1]-1, pipeline.img_size[1])
            left_pts = np.c_[pipeline.left_lane.current_x_fitted, y_pts].astype(np.int32)
            right_pts = np.c_[pipeline.right_lane.current_x_fitted, y_pts].astype(np.int32)
            
            warped = cv2.polylines(warped, [left_pts.reshape((-1,1,2))], isClosed=False, color=(255), thickness=2)
            warped = cv2.polylines(warped, [right_pts.reshape((-1,1,2))], isClosed=False, color=(255), thickness=2)           
       
        return_vars, outimg = proc.scan_find_lane(warped, margin = 120)
        
        if return_vars['left_found']:
            if len(pipeline.left_lane.recent_x_fitted) == 0 or \
                    pipeline.left_lane.sanity_check(return_vars['left_fit_x']):
                l_found = True
                
        elif i == 1:
            pipeline.left_lane.detected = False
        
        if return_vars['right_found']:
            if len(pipeline.right_lane.recent_x_fitted) == 0 or \
                    pipeline.right_lane.sanity_check(return_vars['right_fit_x']):
                r_found = True
                
        elif i == 1:
            pipeline.right_lane.detected = False
            
        if l_found and r_found:
            poly_out = proc.drawPoly(img, return_vars['left_fit_x'], return_vars['fit_y'], 
                                     return_vars['right_fit_x'], return_vars['fit_y'], M=pipeline.Minv)    
            out = proc.addText(poly_out, return_vars['left_curverad'], return_vars['right_curverad'], 
                               return_vars['center_loc'], M=pipeline.Minv)
            
            pipeline.left_lane.update(True, return_vars['left_fit_x'], return_vars['left_fit_eq'], 
                                      return_vars['left_curverad'])
            pipeline.right_lane.update(True, return_vars['right_fit_x'], return_vars['right_fit_eq'], 
                                       return_vars['right_curverad'])
            out = cv2.resize(out,None,fx=0.66, fy=0.66, interpolation = cv2.INTER_AREA)
            return out
        elif i == 1:  
            y_pts = np.linspace(0, pipeline.img_size[1]-1, pipeline.img_size[1])
            xm_per_pix = 3.7 / (pipeline.right_lane.current_x_fitted[-1] - 
                                pipeline.left_lane.current_x_fitted[-1])
            center_loc = (-(pipeline.right_lane.current_x_fitted[-1] 
                        + pipeline.left_lane.current_x_fitted[-1]) // 2 + img.shape[1] // 2) * xm_per_pix
            poly_out = proc.drawPoly(img, pipeline.left_lane.current_x_fitted, y_pts, 
                                     pipeline.right_lane.current_x_fitted, y_pts, M=pipeline.Minv)    
            out = proc.addText(poly_out, pipeline.left_lane.radius_of_curvature, pipeline.right_lane.radius_of_curvature,
                               center_loc, M=pipeline.Minv)
            out = cv2.resize(out,None,fx=0.66, fy=0.66, interpolation = cv2.INTER_AREA)
    return out