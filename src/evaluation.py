import sys
if "" not in sys.path : sys.path.append("")

import numpy as np
from keras.models import load_model


# evaluate loaded model on test data 
def evaluate_model(dataset_test, model_Filename):
    X_test, Y_test = dataset_test
    X_test = np.expand_dims(X_test, axis=3)
    Y_test = np.expand_dims(Y_test, axis=3)
    loaded_model = load_model(model_Filename)
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)
    print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))