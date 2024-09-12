import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
from skimage.io import imread
import matplotlib
from segment_anything import sam_model_registry, SamPredictor

import cv2
import torch

from matplotlib.gridspec import GridSpec
from skimage import transform, color
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.morphology import square, closing
from skimage import filters


# Get upper and lower points of glottis given glottis mask
def get_glottis_data(gmask):
    
    up=[np.inf,np.inf]
    up_x = []
    low=[-1,-1]
    low_x=[]
    
    h,w = np.shape(gmask)
    img = gmask
    for y in range(h):
        for x in range(w):
            val = gmask[y,x]
            if val!=0:
                if y<up[1]:
                    up[1]=y
                up_x.append(x)
        up[0]=np.mean(up_x)
        if up[1]!=np.inf:
            break
            
    for y in range(h-1,-1,-1):
        for x in range(w):
            val = gmask[y,x]
            if val!=0:
                if y>low[1]:
                    low[1]=y
                low_x.append(x)
        low[0]=np.mean(low_x)
        if low[1]!=-1:
            break
    return np.array([up,low])

# get the m,b for orthogonal lines
def get_line_o(m,p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    
    b = p2[1] - m * p2[0]
    # print(b)
    return b

# get m,b for parallel lines
def get_line_p(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    if x2==x1:
        k = -50
    else:
        k = (y2-y1)/(x2-x1)
    b = y2-k*x2
    return k,b

# get x,y and gray value for each line
def get_lists(m,b,img):
    x_list=[]
    y_list=[]
    gray_val = []
    x_min = 0
    x_max = 0
    
    if m<0:
        x_min = max(0,int((255-b)/m)+1)
        x_max = min(256,int((0-b)/m))
        
    elif m>0:
        x_min = max(0,int((0-b)/m))
        x_max = min(256,int((255-b)/m)+1)
    
    for i in range(x_min,x_max):
        x_list.append(i)
        y = round(i*m+b)
        y_list.append(y)
        gray_val.append(img[y,i])
        
    return x_list,y_list,gray_val

# get the first derivative of gray value
def get_fd(x_list, y_list, gray_val, point):
    dy_dx = np.gradient(gray_val,x_list)    
    dy_dx = LabberRing(dy_dx,x_list)
    return dy_dx

# smooth the first derivative values 
def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re
 
def LabberRing(dy_dx,x_list):
    y = dy_dx
    t = np.linspace(x_list[0], x_list[-1], len(y)) 
    y_av = moving_average(y, 5)
    return y_av

def is_local_minimum(array, index):
    if index <= 0 or index >= len(array) - 1:
        return False
    return array[index] <= array[index - 1] and array[index] <= array[index + 1]

def find_nearest_local_minimum(array, start_index, direction):
    index = start_index
    while 0 <= index < len(array):
        if is_local_minimum(array, index):
            return index
        index += direction
    return None

def find_local_minima(array, p, i):
    p = p-i
    if is_local_minimum(array, p):
        nearest_min = p
    else:
        left_min = find_nearest_local_minimum(array, p - 1, -1)
        right_min = find_nearest_local_minimum(array, p + 1, 1)
        # print(left_min+i,right_min+i)
        if left_min is None and right_min is None:
            return None, None, None
        
        if left_min is None:
            nearest_min = right_min
        elif right_min is None:
            nearest_min = left_min
        else:
            if abs(left_min - p) <= abs(right_min - p):
                nearest_min = left_min
            else:
                nearest_min = right_min
    
    left_of_nearest = find_nearest_local_minimum(array, nearest_min - 1, -1)
    right_of_nearest = find_nearest_local_minimum(array, nearest_min + 1, 1)
    
    if nearest_min!=None:
        nearest_min+=i
    if left_of_nearest!=None:
        left_of_nearest+=i
    if right_of_nearest!=None:
        right_of_nearest+=i
    
    return nearest_min, left_of_nearest, right_of_nearest

def is_local_maximum(array, index):
    if index <= 0 or index >= len(array) - 1:
        return False
    return array[index] >= array[index - 1] and array[index] >= array[index + 1]

def find_nearest_local_maximum(array, start_index, direction):
    index = start_index
    while 0 <= index < len(array):
        if is_local_maximum(array, index):
            return index
        index += direction
    return None

def find_local_maxima(array, p, i):
    p = p-i
    if is_local_maximum(array, p):
        nearest_max = p
    else:
        left_max = find_nearest_local_maximum(array, p - 1, -1)
        right_max = find_nearest_local_maximum(array, p + 1, 1)
        
        if left_max is None and right_max is None:
            return None, None, None
        
        if left_max is None:
            nearest_max = right_max
        elif right_max is None:
            nearest_max = left_max
        else:
            if abs(left_max - p) < abs(right_max - p):
                nearest_max = left_max
            else:
                nearest_max = right_max
    
    left_of_nearest = find_nearest_local_maximum(array, nearest_max - 1, -1)
    right_of_nearest = find_nearest_local_minimum(array, nearest_max + 1, 1)
    
    if nearest_max!=None:
        nearest_max+=i
    if left_of_nearest!=None:
        left_of_nearest+=i
    if right_of_nearest!=None:
        right_of_nearest+=i
    
    return nearest_max, left_of_nearest, right_of_nearest

def find_local_maximum(array, p, i, d):
    p = p-i
    if d=='l':  
        point = find_nearest_local_minimum(array, p -1, -1)

    elif d=='r':
        point = find_nearest_local_maximum(array, p +1, 1)

    if point!=None:
        point+=i    
    return point

def get_lists_v2(d,b,img):
    x_list=[]
    y_list=[]
    gray_val = []
    x_min = 0
    x_max = 0
    
    if d=='h':
        x_min =0
        x_max =256
        for i in range(x_min,x_max):
            x_list.append(i)
            y = b
            y_list.append(y)
            gray_val.append(img[y,i])
        
    elif d=='v':
        y_min = 0
        y_max =256
    
        for i in range(y_min,y_max):
            y_list.append(i)
            x_list.append(b)
            gray_val.append(img[i,b])

    return x_list,y_list,gray_val


def find_local_maximum_v2(array, p, d):
    if d=='u':  
        point = find_nearest_local_minimum(array, p -1, -1)

    elif d=='d':
        point = find_nearest_local_maximum(array, p +1, 1)
        
    return point

def find_edge_points(gmask,b):
    for i in range(256):
        if i == 255:
            p1=(-1,-1)
        else:
            val = gmask[b,i]
            next_val = gmask[b,i+1]
            if val==0 and next_val==1:
                p1 = [i+1,b]
                break
    for i in range(255,-1,-1):
        if i==0:
            p2=(-1,-1)
        else:
            val = gmask[b,i]
            pre_val = gmask[b,i-1]
            if val==0 and pre_val==1:
                p2=[i-1,b]
                break
        
                
    return p1,p2
    
def find_local_maxima_v2(array, p1, p2):
    p1 = p1
    p2 = p2

    left = find_nearest_local_maximum(array, p1, -1)
    right = find_nearest_local_minimum(array, p2, 1)

    return -1, left, right


# calculate the angle of rotation
def calculate_angle(A, B, C):

    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    BA = A - B
    BC = C - B

    dot_product = np.dot(BA, BC)

    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)

    cos_angle = dot_product / (norm_BA * norm_BC)

    angle_radians = np.arccos(cos_angle)

    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# move and rotate the image so that the mid point of the glottis is in the middle of the image
def move_point_to_center_and_rotate(image, point, angle, Is_white=False):
    w,h = 256,256

    center = (w // 2, h // 2)
    
    tx = center[0] - point[0]
    ty = center[1] - point[1]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    
    if Is_white:
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h), borderValue=(0, 255, 255))
    else:
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
   
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if Is_white:
        rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (w, h),borderValue=(0, 255, 255))
    else:
        rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (w, h))
    return rotated_image

# load image
def load_image_v2(img, h, w):
    if len(img.shape) > 2:
        return img_as_ubyte(
            np.clip(
                resize(
                    rgb2gray(img), (h, w)
                ), -1.0, 1.0)
            )
    else:
        return img_as_ubyte(
            np.clip(
                resize(
                    img, (h, w)
                ), -1.0, 1.0)
            ) / 255
    
# move back the image given the mid point and angle
def move_back(image, point, angle):
    h, w = np.shape(image)
    center = (w // 2, h // 2)
   
    reverse_angle = -angle
    rotation_matrix = cv2.getRotationMatrix2D(center, reverse_angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    tx = point[0] - center[0]
    ty = point[1] -center[1]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    
    original_image = cv2.warpAffine(rotated_image, translation_matrix, (w, h))
    
    return original_image


# apply CLAHE to intensity channel of the image
def get_CLAHE(img, gridsize=3,l=3):

    clahe = cv2.createCLAHE(clipLimit=l, tileGridSize=(gridsize,gridsize))

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    bgr_CLAHE = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    Lum_CLAHE = cv2.cvtColor(bgr_CLAHE,cv2.COLOR_BGR2RGB)
    return Lum_CLAHE

# update the bounding box according to the input points
def get_box(input_points, k=10):
    coords = input_points
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    box = np.array([x_min-k,y_min-k,x_max+k,y_max+k])
    return box


# get the bounding box using U-Net mask
def peek_v2(img, net, yolo_model, device, Is_CLAHE=True, threshold=1e-17, N=1):
    net.eval()    
    masks = []
    
    ori=np.copy(img)

    
    width, height = 256,256
    

    img = load_image_v2(img, 512, 256)
    img = img/255
    
    img = torch.as_tensor(img.copy())[None].unsqueeze(0).float().contiguous().to(device)
    with torch.no_grad():
        mask = net(img)
        masks.append(mask)
        mask1 = (torch.sigmoid(mask)>0.5).detach().cpu().float().numpy().squeeze(0).squeeze(0)
        mask2 = (torch.sigmoid(mask)<threshold).detach().cpu().float().numpy().squeeze(0).squeeze(0)

    mask1 = resize(mask1, (height, width))
    mask2 = resize(mask2, (height, width))

    for i in range(256):
        for j in range(256):
            if mask1[i,j]!=1:
                mask1[i,j]=0

    if Is_CLAHE:
        CLAHE = get_CLAHE(ori)
        CLAHE = color.rgb2gray(CLAHE)
        CLAHE = transform.resize(CLAHE, (512,256), anti_aliasing=True)

        CLAHE = torch.as_tensor(CLAHE.copy())[None].unsqueeze(0).float().contiguous().to(device)
        with torch.no_grad():
            mask = net(CLAHE)
            # masks.append(mask)
            CLAHE_mask = (torch.sigmoid(mask)<threshold).detach().cpu().float().numpy().squeeze(0).squeeze(0)
        
        CLAHE_mask = resize(CLAHE_mask, (height, width))
        x1,x2,y1,y2=-1,-1,-1,-1
        if np.sum(mask1)>20:
            yolo_results = yolo_model(ori, size=640).pandas().xyxy   

            
            if not yolo_results[0].empty:
                x1 = int(yolo_results[0]['xmin'].iloc[0])
                y1 = int(yolo_results[0]['ymin'].iloc[0])
                x2 = int(yolo_results[0]['xmax'].iloc[0])
                y2 = int(yolo_results[0]['ymax'].iloc[0])

                    # print(angle)
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                if x2>255:
                    x2=255
                if y2>255:
                    y2=255

                for i in range(y1,y2):
                    for j in range(x1,x2):
                        mask2[i,j] = CLAHE_mask[i,j]
                        
                for h in range(256):
                    for w in range(256):
                        if w<x1 or w>x2 or h<y1 or h>y2:
                            mask1[h,w]=0
                
                
            else:  mask2 = CLAHE_mask

        else:
            mask2 = CLAHE_mask

    # print(yolo_results)
    
    contours,hierarchy =cv2.findContours(np.uint8(mask1),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                      )
    # print(contours)
    if contours!=():
        
        for contour in contours:
            # 创建一个与原图像大小相同的空白图像
            blank_image = np.zeros_like(mask1, dtype=np.uint8)

            # 在空白图像上绘制并填充轮廓
            cv2.drawContours(blank_image, [contour], -1, (255), thickness=cv2.FILLED)
            
            # 计算填充轮廓内的像素点数
            num_pixels = cv2.countNonZero(blank_image)
            if num_pixels<10:
                blank_image = blank_image/255
                for i in range(256):
                    for j in range(256):
                        if blank_image[i,j]!=0:
                            mask1[i,j]=0 
    
 
        
    
    
    gmask=np.copy(mask1)
    
    
    if np.sum(mask1)<2:
       
        mask2 = resize(mask2, (height, width))

        

        thresh = filters.threshold_otsu(mask2)


        mask2 = mask2 > thresh

        mask2[:10, :] = False         
        mask2[-10:, :] = False        
        mask2[:, :10] = False         
        mask2[:, -10:] = False

        
        bw= np.copy(mask2)
        selem = square(1)
        closed = closing(bw, selem)
        mask2 = closed


        contours,hierarchy =cv2.findContours(np.uint8(mask2),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        mask2 = np.uint8(mask2)
        dst = color.gray2rgb(mask2*255)
        image_plus = cv2.drawContours(dst, max_contour, 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,255,255),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )

        

        # mask2, num_labels = measure.label(closed, connectivity=2, return_num=True)
        output_mask = np.zeros((256,256,3))
        img = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        img = color.rgb2gray(img)/255
        
        
        min_x=np.inf
        max_x=0
        min_y=np.inf
        max_y=0
        for y in range(256):
            for x in range(256):
                val = int(img[y,x])
                if val==1:
                    if x<min_x:
                        min_x=x
                    if x>max_x:
                        max_x=x
                    if y<min_y:
                        min_y=y
                    if y>max_y:
                        max_y=y
        # print(min_x,min_y)
        bbox = np.array([min_x-10,min_y-10,max_x+10,max_y+10])
        # image = move_point_to_center_and_rotate(color.gray2rgb(np.copy(resize(color.rgb2gray(ori), (256,256)))),mid, angle)
        image = resize(ori, (256,256))
        image = cv2.rectangle(image, (min_x-10, min_y-10), (max_x+10, max_y+10), (0, 255, 0), 2) 
        
    
        
        
        
        
        return [],img, bbox,-1,-1
    
    else:
        points = get_glottis_data(mask1)

        mid = [(points[0][0]+points[1][0])//2,(points[0][1]+points[1][1])//2]


        
        angle = calculate_angle(points[1], mid, [mid[0],mid[1]+3])
        
        mask1 = mask1.astype(np.uint8)
        mask1 = move_point_to_center_and_rotate(mask1, mid, angle)
        
        contours,hierarchy =cv2.findContours(np.uint8(mask1),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )
        gmask = np.copy(mask1)
        
        if contours!=():
            max_contour = max(contours, key=cv2.contourArea)
            l = np.copy(max_contour)
            left_max = np.min(l[:, 0, 0])
            right_max = np.max(l[:, 0, 0])

            for contour in contours:
                # 创建一个与原图像大小相同的空白图像
                blank_image = np.zeros_like(mask1, dtype=np.uint8)

                # 在空白图像上绘制并填充轮廓
                cv2.drawContours(blank_image, [contour], -1, (255), thickness=cv2.FILLED)

                # 计算填充轮廓内的像素点数
                num_pixels = cv2.countNonZero(blank_image)
                l = np.copy(contour)
                left_c = np.min(l[:, 0, 0])
                right_c = np.max(l[:, 0, 0])
                # 找到最左和最右的 x 值


                ### mask面积太小
                if right_c<left_max or left_c>right_max:
                # if num_pixels<10:
                    blank_image = blank_image/255
                    for i in range(256):
                        for j in range(256):
                            if blank_image[i,j]!=0:
                                mask1[i,j]=0 
                                
       
        
        
        gmask = np.copy(mask1)
        mask1_contours,hierarchy =cv2.findContours(np.uint8(mask1),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )
        glottis_mask=np.copy(mask1)


        


        points=[]

        for c in mask1_contours:
            y,_,x=np.shape(c)
            c = np.reshape(c,(y,x))
            minimum = c[np.argmin(c[:,1])]
            maximum = c[np.argmax(c[:,1])]
            points.append(list(minimum))
            points.append(list(maximum))
        max_point = points[np.argmax(np.array(points)[:,1])]
        min_point = points[np.argmin(np.array(points)[:,1])]



        mask1 = color.gray2rgb(mask1*255)
        cv2.line(mask1, min_point, max_point, (255, 255, 255), 5)  # White line for visibility





        mask1 = color.rgb2gray(mask1/255)
        mask2 = resize(mask2, (height, width))

        thresh = filters.threshold_otsu(mask2)


        mask2 = mask2 > thresh

        mask2[:10, :] = False         
        mask2[-10:, :] = False        
        mask2[:, :10] = False         
        mask2[:, -10:] = False

        rotate_mask2 = np.copy(mask2).astype(np.uint8)
        rotate_mask2 = move_point_to_center_and_rotate(rotate_mask2, mid, angle)
        rotate_mask2 = rotate_mask2.astype(np.uint8)+mask1.astype(np.uint8)

        bw= np.copy(rotate_mask2)
        


        selem = square(1)
        closed = closing(bw, selem)
        rotate_mask2 = closed
        
        contours,hierarchy =cv2.findContours(np.uint8(rotate_mask2),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        rotate_masks2 = np.uint8(rotate_mask2)
        dst = color.gray2rgb(rotate_mask2*255)
        image_plus = cv2.drawContours(dst, max_contour, 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,0,0),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )


        candidates_contours = []
        p_mid = ((max_point[0]+min_point[0])//2,(max_point[1]+min_point[1])//2)
        p_mid = tuple(map(int, p_mid))
        p_max = tuple(map(int,max_point))
        for c in contours:
            if cv2.pointPolygonTest(c, p_mid, False)!=-1:
                candidates_contours.append(c)



        max_contour = max(candidates_contours, key=cv2.contourArea)

        dst = color.gray2rgb(rotate_mask2*255)
        final_mask = cv2.drawContours(dst, [max_contour], 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,0,0),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )



        output_mask = np.zeros((256,256,3))
        img = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        img = color.rgb2gray(img)/255
        


        min_x=np.inf
        max_x=0
        min_y=np.inf
        max_y=0
        for y in range(256):
            for x in range(256):
                val = int(img[y,x])
                if val==1:
                    if x<min_x:
                        min_x=x
                    if x>max_x:
                        max_x=x
                    if y<min_y:
                        min_y=y
                    if y>max_y:
                        max_y=y

        bbox = np.array([min_x-10,min_y-10,max_x+10,max_y+10])
    return gmask,img, bbox, mid, angle


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def SAM(sam,img,mask,input_point,input_label,box,show_SAM=False, multi_masks=False):    
    predictor = SamPredictor(sam)

    predictor.set_image(img)

    image = img

    if np.shape(mask)[0]!=0:
        

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box = box,
            mask_input = mask[None,:,:],
            multimask_output=multi_masks,
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box = box,
            multimask_output=multi_masks,
        )
 
    
    if show_SAM:
        plt.figure(figsize=(15,5))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.subplot(1,3,i+1)
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_box(box,plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.axis('off')

        plt.show() 
    return masks, scores, logits

def get_sam_masks_unet_with_yolo_lr_v7(img, gtmask, sam, real_gmask, bbox, mid_p, angle, gmask, yolo_model, iterations=2, show_peek=False, show_SAM=False, show_dif=False, multi_masks=False):
    mask = gtmask
    img = np.uint8(img)
    gray = rgb2gray(img)
    h,w = 256,256

    if np.sum(real_gmask)>2:
        if angle!=-1:
            rotate_img = move_point_to_center_and_rotate(img,mid_p,angle,True)
            yolo_results = yolo_model(rotate_img, size=640).pandas().xyxy
        else:
            yolo_results = yolo_model(img, size=640).pandas().xyxy
        

        x1,x2,y1,y2=0,0,0,0
        if not yolo_results[0].empty:
            x1 = yolo_results[0]['xmin'].iloc[0]
            y1 = yolo_results[0]['ymin'].iloc[0]
            x2 = yolo_results[0]['xmax'].iloc[0]
            y2 = yolo_results[0]['ymax'].iloc[0]

            box = [x1,y1,x2,y2]

            bbox[1]=(y1+bbox[1])//2
            bbox[3]=(y2+bbox[3])//2
            bbox[0]=(x1+bbox[0])//2
            bbox[2]=(x2+bbox[2])//2

    if np.shape(gmask)[0]==0:
        input_point=np.array([[(bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2]])
        input_label = np.array([1])
        box = np.array(bbox)

        for i in range(iterations-1):
            if i==0:
                sam_masks, scores, logits=SAM(sam, img,[],input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
            else:
                sam_masks, scores, logits=SAM(sam, img,mask_input,input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
        if iterations==1:
            sam_masks, scores, logits=SAM(sam, img,[],input_point,input_label,box,False,multi_masks)
 
        else:
            sam_masks, scores, logits=SAM(sam, img,mask_input,input_point,input_label,None,False,multi_masks)
            
        sam_mask=sam_masks[0].astype(np.uint8)
        
        contours,hierarchy =cv2.findContours(sam_mask,                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        output_mask = np.zeros((256,256,3))
        sam_mask = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        sam_mask = color.rgb2gray(sam_mask)/255
        sam_mask = sam_mask.astype(np.uint8)


        h,w = 256,256
        
        sam_mask_l=np.copy(sam_mask)
        sam_mask_r= np.copy(sam_mask)
        
        for i in range(h):
            left_boundary=-1
            right_boundart=-1
            mid=-1
            for j in range(w):
                if sam_mask[i,j]==1:
                    left_boundary=j
                    break
            for j in range(w-1,-1,-1):
                if sam_mask[i,j]==1:
                    right_boundary=j
                    break
            if left_boundary!=-1 and right_boundary!=-1:
                mid = (left_boundary+right_boundary)//2
                for j in range(left_boundary,right_boundary+1):
                    if j<mid+1:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0       
    else:
    
         
        x1=bbox[0]
        x2=bbox[2]
        y1=bbox[1]
        y2=bbox[3]

        rotate_gray = move_point_to_center_and_rotate(gray,mid_p,angle)
        rotate_img = move_point_to_center_and_rotate(img,mid_p,angle)



        # matplotlib.use('module://matplotlib_inline.backend_inline')

        LoG=rotate_gray

        points = get_glottis_data(gmask)
        if points[0,0]==0 and points[0,1]==0:

            return -1,-1

        p_0 = points[0]
        p_100 = points[1]
        p_50 = ((p_0[0]+p_100[0])//2,(p_0[1]+p_100[1])//2)
        p_25 = ((p_50[0]+p_0[0])//2,(p_50[1]+p_0[1])//2)
        p_75 = ((p_50[0]+p_100[0])//2,(p_50[1]+p_100[1])//2)


        b_25 = int(p_25[1])
        b_50 = int(p_50[1])
        b_75 = int(p_75[1])
        b_100 = int(p_100[1])
        b0 = int(p_25[0])

        x_list_25, y_list_25, log_25 = get_lists_v2('h',b_25,LoG)
        x_list_50, y_list_50, log_50 = get_lists_v2('h',b_50,LoG)
        x_list_75, y_list_75, log_75 = get_lists_v2('h',b_75,LoG)
        x_list_100, y_list_100, log_100 = get_lists_v2('h',b_100,LoG)

        x_list_0, y_list_0, log_0 = get_lists_v2('v',b0,LoG)

        dy_dx_25 = get_fd(x_list_25, y_list_25, log_25, p_25)
        dy_dx_50 = get_fd(x_list_50, y_list_50, log_50, p_50)
        dy_dx_75 = get_fd(x_list_75, y_list_75, log_75, p_75)
        dy_dx_100 = get_fd(x_list_100, y_list_100, log_100, p_100)

        dy_dx_0 = get_fd(y_list_0, x_list_0, log_0, p_100)


        edge_p1_25,edge_p2_25=find_edge_points(gmask,b_25)
        edge_p1_50,edge_p2_50=find_edge_points(gmask,b_50)
        edge_p1_75,edge_p2_75=find_edge_points(gmask,b_75)


        if edge_p1_25[0]!=-1:
            max_25 = find_local_maxima_v2(dy_dx_25,int(edge_p1_25[0]),int(edge_p2_25[0]))
        else:
            max_25 = find_local_maxima(dy_dx_25,int(p_25[0]),x_list_25[0])

        if edge_p1_50[0]!=-1:
            max_50 = find_local_maxima_v2(dy_dx_50,int(edge_p1_50[0]),int(edge_p2_50[0]))
        else:
            max_50 = find_local_maxima(dy_dx_50,int(p_50[0]),x_list_50[0])

        if edge_p1_75[0]!=-1:    
            max_75 = find_local_maxima_v2(dy_dx_75,int(edge_p1_75[0]),int(edge_p2_75[0]))
        else:
            max_75 = find_local_maxima(dy_dx_75,int(p_75[0]),x_list_75[0])

        max_100 = find_local_maxima(dy_dx_100,int(p_100[0]),x_list_100[0])


        max_00 = find_local_maximum_v2(dy_dx_0, y1,'d')
        max_0 = find_local_maximum_v2(dy_dx_0, y2,'u')

        if max_0 == None or max_00 == None:
            max_0 = y2
            max_00 = y1

        if max_0!=-1:
            p_125 = (p_25[0],(p_100[1]+max_0)//2)
            b_125 = int(p_125[1])
            x_list_125, y_list_125, log_125 = get_lists_v2('h',b_125,LoG)
            dy_dx_125 = get_fd(x_list_125, y_list_125, log_125, p_125)

            max_125 = find_local_maxima(dy_dx_125,int(p_125[0]),x_list_125[0])
        else:
            max_125=(-1,-1,-1)

        if max_00!=-1:
            p_m25 = (p_25[0],(p_0[1]+max_00)//2)
            b_m25 = int(p_m25[1])
            x_list_m25, y_list_m25, log_m25 = get_lists_v2('h',b_m25,LoG)
            dy_dx_m25 = get_fd(x_list_m25, y_list_m25, log_m25, p_m25)

            max_m25 = find_local_minima(dy_dx_m25,int(p_m25[0]),x_list_m25[0])
        else:
            max_m25=(-1,-1,-1)

        s1=[max_25[1],b_25]
        s2=[max_25[2],b_25]


        s3=[max_50[1],b_50]

        s4=[max_50[2],b_50]


        s5=[max_75[1],b_75]

        s6=[max_75[2],b_75]


        s7=[max_100[1],b_100]

        s8=[max_100[2],b_100]

        if max_0!=-1:
            s9=[b0,max_0]
        else:
            s9=[-1,-1]

        if max_00!=-1:

            s10=[b0,max_00]
        else:
            s10=[-1,-1]

        if max_125[0]!=-1:
            s11=[max_125[1],b_125]

            s12=[max_125[2],b_125]
        else:
            s11=[-1,-1]
            s12=[-1,-1]

        if max_m25[0]!=-1:
            s13=[max_m25[1],b_m25]

            s14=[max_m25[2],b_m25]
        else:
            s13=[-1,-1]
            s14=[-1,-1]

        mid = ((points[0][0]+points[1][0])//2,(points[0][1]+points[1][1])//2)
        input_p = [s7,s8,s9,s10,s11,s12,s13,s14,s1,s2,s3,s4,s5,s6,points[0],points[1],mid]
 
        for i in range(12):
            point = input_p[i]
            if point[0]==None or point[1]==None:
                input_p[i]=[-1,-1]
                continue
            if point[0]<x1 or point[0]>x2:
                input_p[i]=[-1,-1]




        input_point=[]

        for point in input_p:
            if point[0]!=-1:
                input_point.append([point[0],point[1]])


        box = bbox

        input_point = input_point[8:]
        input_label = [1 for i in range(len(input_point))]


        input_point = np.array(input_point)
        input_label = np.array(input_label)

        
        for i in range(iterations-1):
            # print(1)
            if i==0:
                sam_masks, scores, logits=SAM(sam, rotate_img,[],input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
            else:
                sam_masks, scores, logits=SAM(sam, rotate_img,mask_input,input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
        if iterations==1:
            sam_masks, scores, logits=SAM(sam, rotate_img,[],input_point,input_label,box,False,multi_masks)

        else:
            sam_masks, scores, logits=SAM(sam, rotate_img,mask_input,input_point,input_label,None,False,multi_masks)

        sam_mask=sam_masks[0].astype(np.uint8)


        contours,hierarchy =cv2.findContours(sam_mask,                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        output_mask = np.zeros((256,256,3))
        sam_mask = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        sam_mask = rgb2gray(sam_mask)/255

        output_masks=[]
        DCs=[]


        sam_mask=sam_mask.astype(np.uint8)
        output_masks.append(move_back(sam_mask,mid_p,angle))

        if np.shape(real_gmask)[0]!=0:
            for i in range(h):
                for j in range(w):
                    if sam_mask[i,j]!=0 and real_gmask[i,j]!=0:
                        sam_mask[i,j]=0

        sam_mask_l = np.copy(sam_mask)
        sam_mask_r = np.copy(sam_mask)


        k1=-np.inf
        k2=-np.inf
        k1=(points[1][1]-128)/(points[1][0]-128)
        k2 = (points[0][1]-128)/(points[0][0]-128)

        if k1 ==-np.inf or k1 == np.inf:
            for i in range(128,256):
                boundary = 128
                for j in range(256):
                    if j<=boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0

        else:
            b1 = -k1*points[1][0]+points[1][1]

            # print(k)
            for i in range(128,256):
                boundary = round((i-b1)/k1)
                for j in range(256):
                    if j<boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0


        if k2 ==-np.inf or k2 == np.inf:
            for i in range(128):
                boundary = 128
                for j in range(256):
                    if j<=boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0

        else:
            b2 = -k2*points[0][0]+points[0][1]

            # print(k)
            for i in range(128):
                boundary = round((i-b2)/k2)
                for j in range(256):
                    if j<boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0
                        
        sam_mask=move_back(sam_mask,mid_p,angle)
        sam_mask_l=move_back(sam_mask_l,mid_p,angle)
        sam_mask_r=move_back(sam_mask_r,mid_p,angle)
        gmask=move_back(gmask,mid_p,angle)
    if np.shape(gmask)[0]==0:
        gmask=np.zeros((256,256))
    predict = np.zeros((256,256,3))
    for i in range(256):
        for j in range(256):
            if gmask[i,j]==1:
                predict[i,j]=[255,0,0]
            elif sam_mask[i,j]==1:
                predict[i,j]=[0,255,0]
    

    h,w = 256,256

    if np.shape(gtmask)[0]!=0:

        GT = 0
        NN = 0
        GT_NN = 0

        GT_l= 0
        P_l = 0
        GT_P_l = 0
        GT_r= 0
        P_r = 0
        GT_P_r = 0
        GT_g= 0
        P_g = 0
        GT_P_g = 0

        pred_l= sam_mask_l
        pred_r = sam_mask_r
        pred = sam_mask
        
        ori=np.zeros([h,w])
        
        for i in range(h):
            for j in range(w):
                if mask[i,j]!=0 and mask[i,j]!=1:
                    ori[i,j]=1    

        for i in range(h):
            for j in range(w):
                if ori[i,j]==1 or pred[i,j]==1:
                    if ori[i,j]==1 and pred[i,j]==1:
                        GT+=1
                        NN+=1
                        GT_NN+=1
                    elif ori[i,j]==1 and sam_mask[i,j]!=1:
                        GT+=1
                    elif ori[i,j]!=1 and sam_mask[i,j]==1:
                        NN+=1
                if pred_l[i,j]==1 or mask[i,j]==2:
                    if pred_l[i,j]==1:
                        P_l+=1
                    if mask[i,j]==2:
                        GT_l+=1
                    if pred_l[i,j]==1 and mask[i,j]==2:
                        GT_P_l+=1
                if pred_r[i,j]==1 or mask[i,j]==3:
                    if pred_r[i,j]==1:
                        P_r+=1
                    if mask[i,j]==3:
                        GT_r+=1
                    if pred_r[i,j]==1 and mask[i,j]==3:
                        GT_P_r+=1
                if np.shape(gmask)[0]!=0:
                    if gmask[i,j]==1 or mask[i,j]==1:
                        if gmask[i,j]==1:
                            P_g+=1
                        if mask[i,j]==1:
                            GT_g+=1
                        if gmask[i,j]==1 and mask[i,j]==1:
                            GT_P_g+=1




        pred_DC_arr_l=[]
        pred_DC_arr_r=[]
        pred_DC_arr_g=[]
        DCs=[]
        output_masks=[sam_mask]

        DC = (2*GT_NN+2.2204*1e-16)/((NN)+(GT)+2.2204*1e-16)
        DCs.append(DC)
        DC = (2*GT_P_l+2.2204*1e-16)/((P_l)+(GT_l)+2.2204*1e-16)
        pred_DC_arr_l.append(DC)
        DC = (2*GT_P_r+2.2204*1e-16)/((P_r)+(GT_r)+2.2204*1e-16)
        pred_DC_arr_r.append(DC)
        DC = (2*GT_P_g+2.2204*1e-16)/((P_g)+(GT_g)+2.2204*1e-16)
        pred_DC_arr_g.append(DC)
        
        return output_masks, DCs, pred_DC_arr_l, pred_DC_arr_r, pred_DC_arr_g

    # if show_dif:
    #     plt.figure(figsize=(15,5))
    #     plt.subplot(141)
    #     plt.imshow(img)
    #     show_mask(sam_mask, plt.gca())
    #     plt.subplot(142)
    #     plt.imshow(sam_mask,cmap='gray')
    #     plt.title('SAM Masks')
    #     if np.shape(gmask)[0]!=0:
    #         plt.subplot(143)
    #         plt.imshow(mask,cmap='gray')
    #         plt.title('Ground Truth')
    #     plt.subplot(144)
    #     plt.imshow(sam_mask_l+2*sam_mask_r,cmap='gray')
    #     plt.title('Predict')
    #     plt.show()

    return predict

def box_prompt_extraction(img, gmask, vf_mask, clahe_mask, bbox, Is_CLAHE=True):
    ori=np.copy(img)
    width, height = 256,256
    mask1 = resize(gmask, (height, width))
    mask2 = resize(vf_mask, (height, width))
    mask3 = resize(clahe_mask, (height, width))

    mask1 = np.uint8(gmask)
    mask2 = np.uint8(vf_mask)
    CLAHE_mask = np.uint8(clahe_mask)
    x1,x2,y1,y2=-1,-1,-1,-1
    if np.sum(mask1)>20:
        if np.shape(bbox)[0]!=0:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>255:
                x2=255
            if y2>255:
                y2=255
            for i in range(y1,y2):
                for j in range(x1,x2):
                    mask2[i,j] = CLAHE_mask[i,j]
                    
            for h in range(256):
                for w in range(256):
                    if w<x1 or w>x2 or h<y1 or h>y2:
                        mask1[h,w]=0  
        else:  mask2 = CLAHE_mask
    else:
        mask2 = CLAHE_mask 
    contours,hierarchy =cv2.findContours(np.uint8(mask1),            
                                         cv2.RETR_LIST,
                                         cv2.CHAIN_APPROX_NONE
                                      )
    if contours!=():
        for contour in contours:
            # 创建一个与原图像大小相同的空白图像
            blank_image = np.zeros_like(mask1, dtype=np.uint8)

            # 在空白图像上绘制并填充轮廓
            cv2.drawContours(blank_image, [contour], -1, (255), thickness=cv2.FILLED)
            
            # 计算填充轮廓内的像素点数
            num_pixels = cv2.countNonZero(blank_image)
            if num_pixels<10:
                blank_image = blank_image/255
                for i in range(256):
                    for j in range(256):
                        if blank_image[i,j]!=0:
                            mask1[i,j]=0
                            
    gmask=np.copy(mask1)
    if np.sum(mask1)<2 and np.sum(mask2)>0:
        mask2[:10, :] = 0         
        mask2[-10:, :] = 0        
        mask2[:, :10] = 0        
        mask2[:, -10:] = 0
        bw= np.copy(mask2)
        selem = square(3)
        closed = closing(bw, selem)
        mask2 = closed
        contours,hierarchy =cv2.findContours(np.uint8(mask2),
                                         cv2.RETR_LIST,
                                         cv2.CHAIN_APPROX_NONE
                                        )
        
        max_contour = max(contours, key=cv2.contourArea)

        mask2 = np.uint8(mask2)
        dst = color.gray2rgb(mask2*255)
        image_plus = cv2.drawContours(dst, max_contour, 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,255,255),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )
        
        output_mask = np.zeros((256,256,3))
        img = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        img = color.rgb2gray(img)/255
        
        min_x=np.inf
        max_x=0
        min_y=np.inf
        max_y=0
        for y in range(256):
            for x in range(256):
                val = int(img[y,x])
                if val==1:
                    if x<min_x:
                        min_x=x
                    if x>max_x:
                        max_x=x
                    if y<min_y:
                        min_y=y
                    if y>max_y:
                        max_y=y
        bbox = np.array([min_x-10,min_y-10,max_x+10,max_y+10])
        image = resize(ori, (256,256))
        image = cv2.rectangle(image, (min_x-10, min_y-10), (max_x+10, max_y+10), (0, 255, 0), 2) 

        return [],img, bbox,-1,-1
    
    elif np.sum(mask1)<2 and np.sum(mask2)==0:
        return [],[],[],-1,-1
    
    
    else:
        points = get_glottis_data(mask1)
        mid = [(points[0][0]+points[1][0])//2,(points[0][1]+points[1][1])//2]
        angle = calculate_angle(points[1], mid, [mid[0],mid[1]+3])
        
        mask1 = mask1.astype(np.uint8)
        mask1 = move_point_to_center_and_rotate(mask1, mid, angle)
        
        contours,hierarchy =cv2.findContours(np.uint8(mask1),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )
        gmask = np.copy(mask1)
        
        if contours!=():
            max_contour = max(contours, key=cv2.contourArea)
            l = np.copy(max_contour)
            left_max = np.min(l[:, 0, 0])
            right_max = np.max(l[:, 0, 0])

            for contour in contours:
                # 创建一个与原图像大小相同的空白图像
                blank_image = np.zeros_like(mask1, dtype=np.uint8)

                # 在空白图像上绘制并填充轮廓
                cv2.drawContours(blank_image, [contour], -1, (255), thickness=cv2.FILLED)

                # 计算填充轮廓内的像素点数
                num_pixels = cv2.countNonZero(blank_image)
                l = np.copy(contour)
                left_c = np.min(l[:, 0, 0])
                right_c = np.max(l[:, 0, 0])
                # 找到最左和最右的 x 值


                ### mask面积太小
                if right_c<left_max or left_c>right_max:
                # if num_pixels<10:
                    blank_image = blank_image/255
                    for i in range(256):
                        for j in range(256):
                            if blank_image[i,j]!=0:
                                mask1[i,j]=0 
                                
        gmask = np.copy(mask1)
        mask1_contours,hierarchy =cv2.findContours(np.uint8(mask1),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )
        glottis_mask=np.copy(mask1)

        points=[]

        for c in mask1_contours:
            y,_,x=np.shape(c)
            c = np.reshape(c,(y,x))
            minimum = c[np.argmin(c[:,1])]
            maximum = c[np.argmax(c[:,1])]
            points.append(list(minimum))
            points.append(list(maximum))
        max_point = points[np.argmax(np.array(points)[:,1])]
        min_point = points[np.argmin(np.array(points)[:,1])]



        mask1 = color.gray2rgb(mask1*255)
        cv2.line(mask1, min_point, max_point, (255, 255, 255), 5)  # White line for visibility

        mask1 = color.rgb2gray(mask1/255)
        mask2 = resize(mask2, (height, width))

        thresh = filters.threshold_otsu(mask2)


        mask2 = mask2 > thresh

        mask2[:10, :] = False         
        mask2[-10:, :] = False        
        mask2[:, :10] = False         
        mask2[:, -10:] = False

        rotate_mask2 = np.copy(mask2).astype(np.uint8)
        rotate_mask2 = move_point_to_center_and_rotate(rotate_mask2, mid, angle)
        rotate_mask2 = rotate_mask2.astype(np.uint8)+mask1.astype(np.uint8)

        bw= np.copy(rotate_mask2)
   
        selem = square(1)
        closed = closing(bw, selem)
        rotate_mask2 = closed
        
        contours,hierarchy =cv2.findContours(np.uint8(rotate_mask2),                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        rotate_masks2 = np.uint8(rotate_mask2)
        dst = color.gray2rgb(rotate_mask2*255)
        image_plus = cv2.drawContours(dst, max_contour, 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,0,0),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )


        candidates_contours = []
        p_mid = ((max_point[0]+min_point[0])//2,(max_point[1]+min_point[1])//2)
        p_mid = tuple(map(int, p_mid))
        p_max = tuple(map(int,max_point))
        for c in contours:
            if cv2.pointPolygonTest(c, p_mid, False)!=-1:
                candidates_contours.append(c)



        max_contour = max(candidates_contours, key=cv2.contourArea)

        dst = color.gray2rgb(rotate_mask2*255)
        final_mask = cv2.drawContours(dst, [max_contour], 
                                      -1,                # index of contour to draw, -1 means all
                                      (255,0,0),         # color of contour
                                      1                  # thickness of line, 2 for better visualization
                                     )



        output_mask = np.zeros((256,256,3))
        img = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        img = color.rgb2gray(img)/255

        min_x=np.inf
        max_x=0
        min_y=np.inf
        max_y=0
        for y in range(256):
            for x in range(256):
                val = int(img[y,x])
                if val==1:
                    if x<min_x:
                        min_x=x
                    if x>max_x:
                        max_x=x
                    if y<min_y:
                        min_y=y
                    if y>max_y:
                        max_y=y

        bbox = np.array([min_x-10,min_y-10,max_x+10,max_y+10])
    return gmask,img,bbox,mid,angle


def vf_mask_extraction(img, gtmask, sam, real_gmask, bbox, mid_p, angle, gmask, yolo_model, iterations=2, show_peek=False, show_SAM=False, show_dif=False, multi_masks=False):
    mask = gtmask
    img = np.uint8(img)
    gray = rgb2gray(img)
    h,w = 256,256

    if np.sum(real_gmask)>2:
        if angle!=-1:
            rotate_img = move_point_to_center_and_rotate(img,mid_p,angle,True)
            yolo_results = yolo_model(rotate_img, size=640).pandas().xyxy
        else:
            yolo_results = yolo_model(img, size=640).pandas().xyxy
        

        x1,x2,y1,y2=0,0,0,0
        if not yolo_results[0].empty:
            x1 = yolo_results[0]['xmin'].iloc[0]
            y1 = yolo_results[0]['ymin'].iloc[0]
            x2 = yolo_results[0]['xmax'].iloc[0]
            y2 = yolo_results[0]['ymax'].iloc[0]

            box = [x1,y1,x2,y2]

            bbox[1]=(y1+bbox[1])//2
            bbox[3]=(y2+bbox[3])//2
            bbox[0]=(x1+bbox[0])//2
            bbox[2]=(x2+bbox[2])//2

    if np.shape(gmask)[0]==0:
        input_point=np.array([[(bbox[0]+bbox[2])//2,(bbox[1]+bbox[3])//2]])
        input_label = np.array([1])
        box = np.array(bbox)

        for i in range(iterations-1):
            if i==0:
                sam_masks, scores, logits=SAM(sam, img,[],input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
            else:
                sam_masks, scores, logits=SAM(sam, img,mask_input,input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
        if iterations==1:
            sam_masks, scores, logits=SAM(sam, img,[],input_point,input_label,box,False,multi_masks)
 
        else:
            sam_masks, scores, logits=SAM(sam, img,mask_input,input_point,input_label,None,False,multi_masks)
            
        sam_mask=sam_masks[0].astype(np.uint8)
        
        contours,hierarchy =cv2.findContours(sam_mask,                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        output_mask = np.zeros((256,256,3))
        sam_mask = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        sam_mask = color.rgb2gray(sam_mask)/255
        sam_mask = sam_mask.astype(np.uint8)


        h,w = 256,256
        
        sam_mask_l=np.copy(sam_mask)
        sam_mask_r= np.copy(sam_mask)
        
        for i in range(h):
            left_boundary=-1
            right_boundart=-1
            mid=-1
            for j in range(w):
                if sam_mask[i,j]==1:
                    left_boundary=j
                    break
            for j in range(w-1,-1,-1):
                if sam_mask[i,j]==1:
                    right_boundary=j
                    break
            if left_boundary!=-1 and right_boundary!=-1:
                mid = (left_boundary+right_boundary)//2
                for j in range(left_boundary,right_boundary+1):
                    if j<mid+1:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0       
    else:
    
         
        x1=bbox[0]
        x2=bbox[2]
        y1=bbox[1]
        y2=bbox[3]
        
        rotate_gray = move_point_to_center_and_rotate(gray,mid_p,angle)
        rotate_img = move_point_to_center_and_rotate(img,mid_p,angle)



        # matplotlib.use('module://matplotlib_inline.backend_inline')

        LoG=rotate_gray

        points = get_glottis_data(gmask)
        if points[0,0]==0 and points[0,1]==0:

            return -1,-1

        p_0 = points[0]
        p_100 = points[1]
        p_50 = ((p_0[0]+p_100[0])//2,(p_0[1]+p_100[1])//2)
        p_25 = ((p_50[0]+p_0[0])//2,(p_50[1]+p_0[1])//2)
        p_75 = ((p_50[0]+p_100[0])//2,(p_50[1]+p_100[1])//2)


        b_25 = int(p_25[1])
        b_50 = int(p_50[1])
        b_75 = int(p_75[1])
        b_100 = int(p_100[1])
        b0 = int(p_25[0])

        x_list_25, y_list_25, log_25 = get_lists_v2('h',b_25,LoG)
        x_list_50, y_list_50, log_50 = get_lists_v2('h',b_50,LoG)
        x_list_75, y_list_75, log_75 = get_lists_v2('h',b_75,LoG)
        x_list_100, y_list_100, log_100 = get_lists_v2('h',b_100,LoG)

        x_list_0, y_list_0, log_0 = get_lists_v2('v',b0,LoG)

        dy_dx_25 = get_fd(x_list_25, y_list_25, log_25, p_25)
        dy_dx_50 = get_fd(x_list_50, y_list_50, log_50, p_50)
        dy_dx_75 = get_fd(x_list_75, y_list_75, log_75, p_75)
        dy_dx_100 = get_fd(x_list_100, y_list_100, log_100, p_100)

        dy_dx_0 = get_fd(y_list_0, x_list_0, log_0, p_100)


        edge_p1_25,edge_p2_25=find_edge_points(gmask,b_25)
        edge_p1_50,edge_p2_50=find_edge_points(gmask,b_50)
        edge_p1_75,edge_p2_75=find_edge_points(gmask,b_75)


        if edge_p1_25[0]!=-1:
            max_25 = find_local_maxima_v2(dy_dx_25,int(edge_p1_25[0]),int(edge_p2_25[0]))
        else:
            max_25 = find_local_maxima(dy_dx_25,int(p_25[0]),x_list_25[0])

        if edge_p1_50[0]!=-1:
            max_50 = find_local_maxima_v2(dy_dx_50,int(edge_p1_50[0]),int(edge_p2_50[0]))
        else:
            max_50 = find_local_maxima(dy_dx_50,int(p_50[0]),x_list_50[0])

        if edge_p1_75[0]!=-1:    
            max_75 = find_local_maxima_v2(dy_dx_75,int(edge_p1_75[0]),int(edge_p2_75[0]))
        else:
            max_75 = find_local_maxima(dy_dx_75,int(p_75[0]),x_list_75[0])

        max_100 = find_local_maxima(dy_dx_100,int(p_100[0]),x_list_100[0])


        max_00 = find_local_maximum_v2(dy_dx_0, y1,'d')
        max_0 = find_local_maximum_v2(dy_dx_0, y2,'u')

        if max_0 == None or max_00 == None:
            max_0 = y2
            max_00 = y1

        if max_0!=-1:
            p_125 = (p_25[0],(p_100[1]+max_0)//2)
            b_125 = int(p_125[1])
            x_list_125, y_list_125, log_125 = get_lists_v2('h',b_125,LoG)
            dy_dx_125 = get_fd(x_list_125, y_list_125, log_125, p_125)

            max_125 = find_local_maxima(dy_dx_125,int(p_125[0]),x_list_125[0])
        else:
            max_125=(-1,-1,-1)

        if max_00!=-1:
            p_m25 = (p_25[0],(p_0[1]+max_00)//2)
            b_m25 = int(p_m25[1])
            x_list_m25, y_list_m25, log_m25 = get_lists_v2('h',b_m25,LoG)
            dy_dx_m25 = get_fd(x_list_m25, y_list_m25, log_m25, p_m25)

            max_m25 = find_local_minima(dy_dx_m25,int(p_m25[0]),x_list_m25[0])
        else:
            max_m25=(-1,-1,-1)

        s1=[max_25[1],b_25]
        s2=[max_25[2],b_25]


        s3=[max_50[1],b_50]

        s4=[max_50[2],b_50]


        s5=[max_75[1],b_75]

        s6=[max_75[2],b_75]


        s7=[max_100[1],b_100]

        s8=[max_100[2],b_100]

        if max_0!=-1:
            s9=[b0,max_0]
        else:
            s9=[-1,-1]

        if max_00!=-1:

            s10=[b0,max_00]
        else:
            s10=[-1,-1]

        if max_125[0]!=-1:
            s11=[max_125[1],b_125]

            s12=[max_125[2],b_125]
        else:
            s11=[-1,-1]
            s12=[-1,-1]

        if max_m25[0]!=-1:
            s13=[max_m25[1],b_m25]

            s14=[max_m25[2],b_m25]
        else:
            s13=[-1,-1]
            s14=[-1,-1]

        mid = ((points[0][0]+points[1][0])//2,(points[0][1]+points[1][1])//2)
        input_p = [s7,s8,s9,s10,s11,s12,s13,s14,s1,s2,s3,s4,s5,s6,points[0],points[1],mid]
 
        for i in range(12):
            point = input_p[i]
            if point[0]==None or point[1]==None:
                input_p[i]=[-1,-1]
                continue
            if point[0]<x1 or point[0]>x2:
                input_p[i]=[-1,-1]




        input_point=[]

        for point in input_p:
            if point[0]!=-1:
                input_point.append([point[0],point[1]])


        box = bbox

        input_point = input_point[8:]
        
        if np.shape(input_point)[0]==0:
            input_point= [[(bbox[0]+bbox[2])//2,(bbox[1]+bbox[3]//2)]]
        
        input_label = [1 for i in range(len(input_point))]

        

        input_point = np.array(input_point)
        input_label = np.array(input_label)
        

        for i in range(iterations-1):
            # print(1)
            if i==0:
                sam_masks, scores, logits=SAM(sam, rotate_img,[],input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
            else:
                sam_masks, scores, logits=SAM(sam, rotate_img,mask_input,input_point,input_label,box,False,multi_masks)
                mask_input = logits[0]
        if iterations==1:
            sam_masks, scores, logits=SAM(sam, rotate_img,[],input_point,input_label,box,False,multi_masks)

        else:
            sam_masks, scores, logits=SAM(sam, rotate_img,mask_input,input_point,input_label,None,False,multi_masks)

        sam_mask=sam_masks[0].astype(np.uint8)


        contours,hierarchy =cv2.findContours(sam_mask,                 # if image is boolean you need to convert with np.uint8(img)
                                         cv2.RETR_LIST,       # retrieves all contours without establishing any hierarchy
                                         cv2.CHAIN_APPROX_NONE# lists all contour points, not endpoints of line segments
                                        )

        max_contour = max(contours, key=cv2.contourArea)

        output_mask = np.zeros((256,256,3))
        sam_mask = cv2.drawContours(output_mask, [max_contour], -1, (255,255,255), thickness=cv2.FILLED)
        sam_mask = rgb2gray(sam_mask)/255

        output_masks=[]
        DCs=[]


        sam_mask=sam_mask.astype(np.uint8)
        output_masks.append(move_back(sam_mask,mid_p,angle))

        if np.shape(real_gmask)[0]!=0:
            for i in range(h):
                for j in range(w):
                    if sam_mask[i,j]!=0 and real_gmask[i,j]!=0:
                        sam_mask[i,j]=0

        sam_mask_l = np.copy(sam_mask)
        sam_mask_r = np.copy(sam_mask)


        k1=-np.inf
        k2=-np.inf
        k1=(points[1][1]-128)/(points[1][0]-128)
        k2 = (points[0][1]-128)/(points[0][0]-128)

        if k1 ==-np.inf or k1 == np.inf:
            for i in range(128,256):
                boundary = 128
                for j in range(256):
                    if j<=boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0

        else:
            b1 = -k1*points[1][0]+points[1][1]

            # print(k)
            for i in range(128,256):
                boundary = round((i-b1)/k1)
                for j in range(256):
                    if j<boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0


        if k2 ==-np.inf or k2 == np.inf:
            for i in range(128):
                boundary = 128
                for j in range(256):
                    if j<=boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0

        else:
            b2 = -k2*points[0][0]+points[0][1]

            # print(k)
            for i in range(128):
                boundary = round((i-b2)/k2)
                for j in range(256):
                    if j<boundary:
                        sam_mask_r[i,j]=0
                    else:
                        sam_mask_l[i,j]=0
                        
        sam_mask=move_back(sam_mask,mid_p,angle)
        sam_mask_l=move_back(sam_mask_l,mid_p,angle)
        sam_mask_r=move_back(sam_mask_r,mid_p,angle)
        gmask=move_back(gmask,mid_p,angle)

    if np.shape(gmask)[0]==0:
        real_gmask=np.zeros((256,256))
    else:
        real_gmask = move_back(real_gmask,mid_p,angle)    
    predict = np.zeros((256,256,3))
    for i in range(256):
        for j in range(256):
            if real_gmask[i,j]==1:
                predict[i,j]=[255,0,0]
            # elif sam_mask_l[i,j]==1:
            #     predict[i,j]=[0,255,0]
            # elif sam_mask_r[i,j]==1:
            #     predict[i,j]=[0,0,255]
            elif sam_mask[i,j]==1:
                predict[i,j]=[0,255,0]
    
    h,w = np.shape(gray)

    if np.shape(gtmask)[0]!=0:

        GT = 0
        NN = 0
        GT_NN = 0

        GT_l= 0
        P_l = 0
        GT_P_l = 0
        GT_r= 0
        P_r = 0
        GT_P_r = 0
        GT_g= 0
        P_g = 0
        GT_P_g = 0

        pred_l= sam_mask_l
        pred_r = sam_mask_r
        pred = sam_mask
        
        ori=np.zeros([h,w])
        
        for i in range(h):
            for j in range(w):
                if mask[i,j]!=0 and mask[i,j]!=1:
                    ori[i,j]=1    

        for i in range(h):
            for j in range(w):
                if ori[i,j]==1 or pred[i,j]==1:
                    if ori[i,j]==1 and pred[i,j]==1:
                        GT+=1
                        NN+=1
                        GT_NN+=1
                    elif ori[i,j]==1 and sam_mask[i,j]!=1:
                        GT+=1
                    elif ori[i,j]!=1 and sam_mask[i,j]==1:
                        NN+=1
                if pred_l[i,j]==1 or mask[i,j]==2:
                    if pred_l[i,j]==1:
                        P_l+=1
                    if mask[i,j]==2:
                        GT_l+=1
                    if pred_l[i,j]==1 and mask[i,j]==2:
                        GT_P_l+=1
                if pred_r[i,j]==1 or mask[i,j]==3:
                    if pred_r[i,j]==1:
                        P_r+=1
                    if mask[i,j]==3:
                        GT_r+=1
                    if pred_r[i,j]==1 and mask[i,j]==3:
                        GT_P_r+=1
                if np.shape(gmask)[0]!=0:
                    if gmask[i,j]==1 or mask[i,j]==1:
                        if gmask[i,j]==1:
                            P_g+=1
                        if mask[i,j]==1:
                            GT_g+=1
                        if gmask[i,j]==1 and mask[i,j]==1:
                            GT_P_g+=1




        pred_DC_arr_l=[]
        pred_DC_arr_r=[]
        pred_DC_arr_g=[]
        DCs=[]
        output_masks=[sam_mask]

        DC = (2*GT_NN+2.2204*1e-16)/((NN)+(GT)+2.2204*1e-16)
        DCs.append(DC)
        DC = (2*GT_P_l+2.2204*1e-16)/((P_l)+(GT_l)+2.2204*1e-16)
        pred_DC_arr_l.append(DC)
        DC = (2*GT_P_r+2.2204*1e-16)/((P_r)+(GT_r)+2.2204*1e-16)
        pred_DC_arr_r.append(DC)
        DC = (2*GT_P_g+2.2204*1e-16)/((P_g)+(GT_g)+2.2204*1e-16)
        pred_DC_arr_g.append(DC)
        
        return output_masks, DCs, pred_DC_arr_l, pred_DC_arr_r, pred_DC_arr_g

    # if show_dif:
    #     plt.figure(figsize=(15,5))
    #     plt.subplot(141)
    #     plt.imshow(img)
    #     show_mask(sam_mask, plt.gca())
    #     plt.subplot(142)
    #     plt.imshow(sam_mask,cmap='gray')
    #     plt.title('SAM Masks')
    #     if np.shape(gmask)[0]!=0:
    #         plt.subplot(143)
    #         plt.imshow(mask,cmap='gray')
    #         plt.title('Ground Truth')
    #     plt.subplot(144)
    #     plt.imshow(sam_mask_l+2*sam_mask_r,cmap='gray')
    #     plt.title('Predict')
    #     plt.show()
    
    return predict