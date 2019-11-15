import joblib
import cv2


bounding_boxes=joblib.load('bbox.job')
depth=joblib.load('depth_array.job')
rgb=joblib.load('rgb_array.job')
timestamp=joblib.load('timestamp.job')

# rgb
cv2.imwrite('raw_img.jpeg', rgb)

# bb
for bb in bounding_boxes[0]:
    cv2.rectangle(rgb, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 1)
for bb in bounding_boxes[1]:
    cv2.rectangle(rgb, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)
cv2.imwrite('filtered_boxed_img.jpeg', rgb)

# depth
normalized_depth = cv2.normalize(depth, depth, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('depth_minmaxnorm.png', normalized_depth)

