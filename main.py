'''
Main program
@Author: Core Tech Centre -VTS

To execute simply run:
main.py

To input new user:
main.py --mode "input"

'''

import cv2
import os
from time import time
import numpy as np
import tensorflow as tf
from SSRNET_model import SSR_net, SSR_net_general
import BKNetStyle3 as BKNetStyle

sess = tf.InteractiveSession()
xbk, ybk_, mask = BKNetStyle.Input()
# y_smile_conv, y_gender_conv, y_age_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(xbk)
y_smile_conv, y_emotion_conv, y_gender_conv, phase_train, keep_prob = BKNetStyle.BKNetModel(xbk)
print('Restore model')
saver = tf.train.Saver()
# saver.restore(sess, '/home/ubuntu/FACENET/Demo_all/demo/save/current5/model-age101.ckpt.index')
saver.restore(sess, './save/done/model.ckpt')
print('OK')
#----------------------------------------------------------------

TIMEOUT = 10  # 10 seconds

size = 160
img_size = 64
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1
weight_file_gender = "./pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
weight_file = "./pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
model.load_weights(weight_file)
model_gender = SSR_net_general(img_size, stage_num, lambda_local, lambda_d)()
model_gender.load_weights(weight_file_gender)



def main():
    img_idx = 0
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    try:
        os.mkdir('./img')
    except OSError:
        pass

    print("[INFO] camera sensor warming up...")
    vs = cv2.VideoCapture('./source_video.avi');  # get input from video
    #vs = cv2.VideoCapture(0);  # get input from webcam
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #in_video = cv2.VideoWriter('input.avi', fourcc, 20.0, (640, 480))
    out_video = cv2.VideoWriter('output.avi', fourcc, 10.0, (640, 480))
    skip_frame = 3

    while True:
        t0 = time()
        _, frame = vs.read();

        frame = cv2.flip(frame, 180)
        #in_video.write(frame)

        img_idx = img_idx + 1

        if frame is None:
            continue

        if img_idx == 1 or img_idx % skip_frame == 0:

            t1 = time()

            print("thoi gian detect %0.3fs" % (t1 - t0))

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray_img, 1.1)
            faces = np.empty((len(detected), img_size, img_size, 3))
            ad = 0.25
            img_w, img_h, _ = np.shape(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX


            if len(detected) > 0:
                for i, (x, y, w, h) in enumerate(detected):

                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

                    xw1 = max(int(x1 - ad * w), 0)
                    yw1 = max(int(y1 - ad * h), 0)
                    xw2 = min(int(x2 + ad * w), img_w - 1)
                    yw2 = min(int(y2 + ad * h), img_h - 1)

                    xw2=xw1+yw2-yw1


                    faces[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
                    faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.rectangle(frame, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

                    input_img_copy = frame.copy()
                    input_img_copy = cv2.cvtColor(input_img_copy, cv2.COLOR_BGR2GRAY)
                    img_bk = cv2.resize(input_img_copy[yw1: yw2 + 1, xw1: xw2 + 1], (48, 48))
                    img_bk = (img_bk - 128) / 255.0
                    T = np.zeros([48, 48, 1])
                    T[:, :, 0] = img_bk
                    test_img = []
                    test_img.append(T)
                    test_img = np.asarray(test_img)

                    predict_y_smile_conv = sess.run(y_smile_conv,
                                                feed_dict={xbk: test_img, phase_train: False, keep_prob: 1})
                    predict_y_emotion_conv = sess.run(y_emotion_conv,
                                                  feed_dict={xbk: test_img, phase_train: False, keep_prob: 1})

                    # angry, disgust, fear, happy, sad, surprise and neutral

                    if np.argmax(predict_y_emotion_conv) == 0:
                        emotion_label = 'angry'
                    elif np.argmax(predict_y_emotion_conv) == 1:
                        emotion_label = 'disgust'
                    elif np.argmax(predict_y_emotion_conv) == 2:
                        emotion_label = 'fear'
                    elif np.argmax(predict_y_emotion_conv) == 3:
                        emotion_label = 'happy'
                    elif np.argmax(predict_y_emotion_conv) == 4:
                        emotion_label = 'sad'
                    elif np.argmax(predict_y_emotion_conv) == 5:
                        emotion_label = 'surprise'
                    elif np.argmax(predict_y_emotion_conv) == 6:
                        emotion_label = 'neutral'
                    else:
                        emotion_label = ''

                    if np.argmax(predict_y_smile_conv) == 0:
                        smile_label = ''
                    else:
                        smile_label = 'smile'

                    cv2.putText(frame, smile_label, (xw1, yw1 + 30), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, emotion_label, (xw1, yw1 + 60), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                if len(detected) > 0:
                    # predict ages and genders of the detected faces
                    predicted_ages = model.predict(faces)  # faces must be in 64 x 64
                    predicted_genders = model_gender.predict(faces)  # faces must be in 64 x 64

                    # draw results
                for i, (x, y, w, h) in enumerate(detected):

                    x1 = x
                    y1 = y
                    xw1 = max(int(x1 - ad * w), 0)
                    yw1 = max(int(y1 - ad * h), 0)

                    gender_str = 'male'
                    if predicted_genders[i] < 0.5:
                        gender_str = 'female'

                    label = "{},{}".format(gender_str, int(predicted_ages[i]))

                    cv2.putText(frame, label, (xw1, yw1), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            t2 = time()
            print("thoi gian du doan tuoi,gioi tinh, cuoi, cam xuc %0.3fs" %( t2 - t1))

            print("done in %0.3fs" % (time() - t0))
            cv2.imwrite('img/' + str(img_idx) + '.png', frame)
            out_video.write(frame)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


if __name__ == '__main__':
    main()
