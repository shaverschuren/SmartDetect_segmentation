import sys
if "" not in sys.path : sys.path.append("")

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt


def evaluate_model(dataset_test, loaded_model):
    X_test, Y_test = dataset_test
    X_test = np.expand_dims(X_test, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
def predict_dataset(dataset, model):
    X_test, Y_test = dataset
    X_test = np.expand_dims(X_test, axis=3)
    #Y_test = np.expand_dims(Y_test, axis=3)
    predicted=model.predict(X_test)
    predicted_images=predicted[:,:,:,0]
    return predicted_images
    
def predict_and_plot_image(image, model):    
    img=np.expand_dims(image, axis=[0,3])
    pred=model.predict(img)
    pred_img=pred[0,:,:,0]
    plt.imshow(pred_img, cmap='gray')
    plt.show()

def dice(input_img, test_img):
    k=1
    dice = np.sum(input_img[test_img==k])*2.0 / (np.sum(input_img) + np.sum(test_img))
    print('Dice similarity score is {}'.format(dice))