import numpy as np

def compute_bbox_center(bbox): #get_center_of_bbox
    # Returns the center (x, y) of a bounding box
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2

def compute_bbox_width(bbox):# get_bbox_width
    # Returns the width of a bounding box
    return bbox[2] - bbox[0]

def compute_euclidean_distance(point1, point2): #measure_distance
    # Computes the Euclidean distance between two points
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def compute_xy_displacement(point1, point2): # measure_xy_distance
    # Returns the x and y displacement between two points
    return point1[0] - point2[0], point1[1] - point2[1]

def compute_foot_position(bbox): #get_foot_position
    # Returns the estimated foot position (mid-bottom) of a bounding box
    x1, _, x2, y2 = bbox
    return (x1 + x2) // 2, y2
