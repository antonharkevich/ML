from PIL import Image
import tensorflow as tf
import numpy as np
from os.path import dirname, join
import cv2
import base64
import imutils
from wurlitzer import pipes
#nessecary imports

#execute operations immediately
tf.compat.v1.disable_eager_execution()

image_size = 32
num_of_channels = 1
num_classifiers = 6

#trained model
model_saver = tf.compat.v1.train.import_meta_graph(join(dirname(__file__), "model.ckpt.meta"))


def prediction_numbers(data):
    #convert android bytes to numpy array and to image
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)
    im_cv2 = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    im = Image.fromarray(im_cv2, 'RGB')

    #resize image
    img = im.resize((image_size, image_size), Image.LANCZOS)


    arr2 = np.array(img)
    img2 = Image.fromarray(arr2)
    img  = img2.convert("RGB")

    #convert to grayscale format
    img = np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])
    pixel_depth = 255.0

    #normalization
    im = (np.array(img, dtype='float32') - (pixel_depth / 2)) / (pixel_depth / 2)
    im = Image.fromarray(im.astype('float32'))

    #reshape to numpy array
    img = np.reshape(im,(-1, image_size, image_size, num_of_channels)).astype(np.float32)

    with tf.compat.v1.Session() as sess:
        model_saver.restore(sess, tf.train.latest_checkpoint(dirname(__file__)))
        graph = sess.graph
        one_input = graph.get_tensor_by_name("one_input_placeholder:0")
        softmax1 = graph.get_tensor_by_name("one_prediction_c1:0")
        softmax2 = graph.get_tensor_by_name("one_prediction_c2:0")
        softmax3 = graph.get_tensor_by_name("one_prediction_c3:0")
        softmax4 = graph.get_tensor_by_name("one_prediction_c4:0")
        softmax5 = graph.get_tensor_by_name("one_prediction_c5:0")
        softmax6 = graph.get_tensor_by_name("one_prediction_c6:0")

        #run prediction
        feed_dict = {one_input: img}
        c1, c2, c3, c4, c5, c6 = sess.run(
            [softmax1, softmax2, softmax3, softmax4, softmax5,
             softmax6], feed_dict=feed_dict)
        predictions = [c1, c2, c3, c4, c5, c6]

        all_labels = []

        #one image
        batch_size = predictions[0].shape[0]
        for i in range(batch_size):
            #predict number of digits
            num_digits = np.argmax(predictions[0][i])
            st = str(num_digits)
            for j in range(1, num_classifiers):
                if (j > num_digits):
                    break
                #predict every digit
                st = st + str(np.argmax(predictions[j][i]))
            all_labels.append(st)

        prediction_str = all_labels

        #return digits without number of digits
        return prediction_str[0][1:]


def prediction_numbers_experimental(data):

    #read image bytes from android
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data, np.uint8)
    im_cv2 = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


    if im_cv2.dtype != "uint8" and im_cv2.dtype != "float32":
        return "Can't Detect"

    with pipes() as (out, err):
        try:
            #resize image
            image = imutils.resize(im_cv2, width=500)

            #convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #reduce noise
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            #detect edges
            edged = cv2.Canny(gray, 170, 200)

            #find rectangle contours region
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
            NumberPlateCnt = None #we currently have no Number plate contour

            # loop over our contours to find the best possible approximate contour of number plate
            count = 0
            for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:  # Select the contour with 4 corners
                        NumberPlateCnt = approx #This is our approx Number Plate Contour
                        break

            if NumberPlateCnt is None:
                 return "Can't Detect"

            mask = np.zeros_like(image) # Create mask where white is what we want, black otherwise

            #crop image with finded contour
            (y, x, _) = np.where(mask == 255)
            (topy, topx) = (np.min(y), np.min(x))
            (bottomy, bottomx) = (np.max(y), np.max(x))
            image = image[topy:bottomy+1, topx:bottomx+1]

        except Exception:
            return "Can't Detect"

    im = Image.fromarray(image, 'RGB')

    #resize image to 32*32
    img = im.resize((image_size, image_size), Image.LANCZOS)


    arr2 = np.array(img)
    img2 = Image.fromarray(arr2)
    img  = img2.convert("RGB")

    #convert to grayscale format
    img = np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])
    pixel_depth = 255.0

    #normalization
    im = (np.array(img, dtype='float32') - (pixel_depth / 2)) / (pixel_depth / 2)
    im = Image.fromarray(im.astype('float32'))

    #reshape to numpy array
    img = np.reshape(im,(-1, image_size, image_size, num_of_channels)).astype(np.float32)

    with tf.compat.v1.Session() as sess:
        model_saver.restore(sess, tf.train.latest_checkpoint(dirname(__file__)))
        graph = sess.graph
        one_input = graph.get_tensor_by_name("one_input_placeholder:0")
        softmax1 = graph.get_tensor_by_name("one_prediction_c1:0")
        softmax2 = graph.get_tensor_by_name("one_prediction_c2:0")
        softmax3 = graph.get_tensor_by_name("one_prediction_c3:0")
        softmax4 = graph.get_tensor_by_name("one_prediction_c4:0")
        softmax5 = graph.get_tensor_by_name("one_prediction_c5:0")
        softmax6 = graph.get_tensor_by_name("one_prediction_c6:0")

        #run prediction
        feed_dict = {one_input: img}
        c1, c2, c3, c4, c5, c6 = sess.run(
            [softmax1, softmax2, softmax3, softmax4, softmax5,
             softmax6], feed_dict=feed_dict)
        predictions = [c1, c2, c3, c4, c5, c6]

        all_labels = []

        #one image
        batch_size = predictions[0].shape[0]
        for i in range(batch_size):
            #predict number of digits
            num_digits = np.argmax(predictions[0][i])
            st = str(num_digits)
            for j in range(1, num_classifiers):
                if (j > num_digits):
                    break
                #predict every digit
                st = st + str(np.argmax(predictions[j][i]))
            all_labels.append(st)

        prediction_str = all_labels

        #return digits without number of digits
        return prediction_str[0][1:]

#example of use python in android
def show_msg(a):
    return a
