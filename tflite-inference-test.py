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
        ret = np.array([ x / 255 for x in list(img.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
    elif (channels == 1):
        rgb_weights = [0.2989, 0.5870, 0.1140]
        img_grayscale = np.dot(img[...,:3], rgb_weights)
        ret = np.array([ x / 255 for x in list(img_grayscale.flatten()) ]).astype(np.float32).reshape((1, width, height, channels))
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

img = cv2.imread(os.path.join(dir_path, 'out/train/000000000000.jpg'))
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

for i in range(len(scores)):
    if ((scores[i] >= minimum_confidence_rating) and (scores[i] <= 1.0)):
        xmin = float(xyxy[0][i])
        ymin = float(xyxy[1][i])
        xmax = float(xyxy[2][i])
        ymax = float(xyxy[3][i])

        # Who in their right min has decided to do ymin,xmin,ymax,xmax ?
        bbox = [ymin, xmin, ymax, xmax]

        rects.append(bbox)
        labels.append(int(classes[i]))
        score_res.append(float(scores[i]))

for i in range(0, len(labels)):
    # if i != 0: continue

    bb = rects[i]
    [ymin, xmin, ymax, xmax] = bb

    xmin = xmin if xmin > 0 else 0
    xmax = xmax if xmax < 1 else 1
    ymin = ymin if ymin > 0 else 0
    ymax = ymax if ymax < 1 else 1

    xmin = int(xmin * img.shape[0])
    xmax = int(xmax * img.shape[0])
    ymin = int(ymin * img.shape[1])
    ymax = int(ymax * img.shape[1])

    print('label', labels[i], 'confidence', score_res[i], 'x', xmin, 'y', ymin, 'w', xmax-xmin, 'h', ymax-ymin)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

cv2.imwrite(os.path.join(dir_path, 'out.png'), img)

# print('rects', rects)
# print('labels', labels)
# print('score_res', score_res)
