import numpy as np

def iou(bb_test, bb_gt, box_mode='xyxy'):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    if box_mode=='xyxy':
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    else:
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[0]+bb_test[2], bb_gt[0]+bb_gt[2])
        yy2 = np.minimum(bb_test[1]+bb_test[3], bb_gt[1]+bb_gt[3])
        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)
        wh = w * h
        o = wh / (bb_test[2] * bb_test[3]
                + bb_gt[2] * bb_gt[3] - wh)

    return o

def linear_assignment(cost_matrix):
    # lap is the fastest package on the linear assignment problem.
    """
    Solve the linear assignment problem.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))