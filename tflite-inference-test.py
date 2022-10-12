import tensorflow as tf
import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def process_input(input_details, data):
    """Prepares an input for inference, quantizing if necessary.

    Args:
        input_details: The result of calling interpreter.get_input_details()
        data (numpy array): The raw input data

    Returns:
        A tensor object representing the input, quantized if necessary
    """
    if input_details[0]['dtype'] is np.int8:
        scale = input_details[0]['quantization'][0]
        zero_point = input_details[0]['quantization'][1]
        data = (data / scale) + zero_point
        data = np.around(data)
        data = data.astype(np.int8)
    return tf.convert_to_tensor(data)

def get_features_from_img(interpreter, img):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    count, width, height, channels = input_shape

    # if channels == width of the image, then we are dealing with channel/width/height
    # instead of height/width/channel
    is_nchw = channels == img.shape[1]
    if (is_nchw):
        count, channels, width, height = input_shape

    if (channels == 3):
        ret = np.array([ x for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        ret = np.array([ x for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    else:
        raise ValueError('Unknown depth for image')

    # transpose the image if required
    if (is_nchw):
        ret = np.transpose(ret, (0, 3, 1, 2))

    return ret

def invoke(interpreter, item, specific_input_shape):
    """Invokes the Python TF Lite interpreter with a given input
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    item_as_tensor = process_input(input_details, item)
    if specific_input_shape:
        item_as_tensor = tf.reshape(item_as_tensor, specific_input_shape)
    # Add batch dimension
    item_as_tensor = tf.expand_dims(item_as_tensor, 0)
    interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output, output_details

interpreter = tf.lite.Interpreter(model_path=os.path.join(dir_path, "out/model.tflite"))
interpreter.allocate_tensors()

img = cv2.imread(os.path.join(dir_path, 'out/train/000000000244.jpg'))
print('img shape', img.shape)

input_data = get_features_from_img(interpreter, img)
print('input_data', input_data.shape)

output, output_details = invoke(interpreter, input_data, list(input_data.shape[1:]))
print('output_details', output_details)

output0 = interpreter.get_tensor(output_details[0]['index'])
# output1 = interpreter.get_tensor(output_details[1]['index'])
print('output0.shape', output0.shape)
# print('output1.shape', output1.shape)

def yolov5_class_filter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

def yolov5_detect(output_data):  # input = interpreter, output is boxes(xyxy), classes, scores
    output_data = output_data[0]                # x(1, 25200, 7) to x(25200, 7)
    boxes = np.squeeze(output_data[..., :4])    # boxes  [25200, 4]
    scores = np.squeeze( output_data[..., 4:5]) # confidences  [25200, 1]
    classes = yolov5_class_filter(output_data[..., 5:]) # get classes
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
    xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]

    return xyxy, classes, scores  # output is boxes(x,y,x,y), classes(int), scores(float) [predictions length]

xyxy, classes, scores = yolov5_detect(output0) #boxes(x,y,x,y), classes(int), scores(float) [25200]

rects = []
labels = []
score_res = []

minimum_confidence_rating = 0.3

# for i in range(len(scores)):
#     if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
#         xmin = float(xyxy[0][i])
#         ymin = float(xyxy[1][i])
#         xmax = float(xyxy[2][i])
#         ymax = float(xyxy[3][i])

#         # Who in their right min has decided to do ymin,xmin,ymax,xmax ?
#         bbox = [ymin, xmin, ymax, xmax]

#         rects.append(bbox)
#         labels.append(int(classes[i]))
#         score_res.append(float(scores[i]))

# for i in range(0, len(labels)):
#     # if i != 0: continue

#     bb = rects[i]
#     [ymin, xmin, ymax, xmax] = bb

#     xmin = xmin if xmin > 0 else 0
#     xmax = xmax if xmax < 1 else 1
#     ymin = ymin if ymin > 0 else 0
#     ymax = ymax if ymax < 1 else 1

#     xmin = int(xmin * img.shape[0])
#     xmax = int(xmax * img.shape[0])
#     ymin = int(ymin * img.shape[1])
#     ymax = int(ymax * img.shape[1])

#     print('label', labels[i], 'confidence', score_res[i], 'x', xmin, 'y', ymin, 'w', xmax-xmin, 'h', ymax-ymin)

#     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

# cv2.imwrite(os.path.join(dir_path, 'out.png'), img)


def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    nms_method = multiclass_nms_class_agnostic
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    dets = []
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

print('output[0]', output0.shape, 'list', [ img.shape[0], img.shape[1] ])
predictions = demo_postprocess(output0, tuple([ img.shape[0], img.shape[1] ]))[0]

boxes = predictions[:, :4]
scores = predictions[:, 4:5] * predictions[:, 5:]

boxes_xyxy = np.ones_like(boxes)
boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.

dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
print('dets', dets)

final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
for i in range(0, len(final_boxes)):
    box = final_boxes[i]

    xmin = int(box[0])
    ymin = int(box[1])
    xmax = int(box[2])
    ymax = int(box[3])

    color = (255, 0, 0) if final_cls_inds[i] == 0.0 else (0, 255, 0)

    print('label', final_cls_inds[i], 'confidence', final_scores[i], 'x', xmin, 'y', ymin, 'w', xmax-xmin, 'h', ymax-ymin)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

cv2.imwrite(os.path.join(dir_path, 'out.png'), img)

# print('rects', rects)
# print('labels', labels)
# print('score_res', score_res)
