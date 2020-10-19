import cv2
import os
import tensorflow as tf
import numpy as np
from pynput.keyboard import Key, Controller

c = Key.space
monitor_height = 1080
monitor_width = 1920
keyboard = Controller()
def start():
    cap = cv2.VideoCapture(0)
    model = tf.keras.models.load_model("Model/rps2.h5")
    while True:
        ret, img = cap.read()
        keyboard.press(Key.space)
        predict_image = img.copy()
        predict_image = cv2.flip(predict_image, 1)
        height, width = predict_image.shape[:2]
        x1, y1 = int(width * 0.25), int(height * 0.25)
        x2, y2 = int(width * 0.75), int(height * 0.8)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        predict_image = img[y1:y2, x1:x2]
        cv2.imshow("Hand", cv2.flip(img, 1))
        cv2.moveWindow("Hand", int(monitor_width / 2), int(monitor_height / 2))
        predict_image = cv2.resize(predict_image, (150, 150), interpolation=cv2.INTER_AREA)
        cv2.imwrite("a.png", predict_image)
        predict_image = tf.keras.preprocessing.image.img_to_array(predict_image)
        predict_image = np.expand_dims(predict_image / 2, axis=0)
        prediction = model.predict(predict_image)

        if prediction[0][0] == 1:

            c = Key.up
            print("Paper:UP")
        elif prediction[0][1] == 1:
            c = Key.down
            print("Rock:DOWN")
        keyboard.release(Key.space)
        keyboard.press(c)
        key = cv2.waitKey(60)
        if key == 27:
            break
        keyboard.release(c)

    keyboard.release(c)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start()
