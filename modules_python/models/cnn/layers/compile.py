import tensorflow as tf
import numpy as np
from modules_python.image_processing.data_aug import data_augmenter_v2


def compile(
        model ,
        Loss : str = "cce", 
        Op   : str = "sgd", 
        scoring = ['accuracy']
        ):
   
    model.compile(
        loss=tf.losses.CategoricalCrossentropy() if Loss == "cce" else tf.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.optimizers.SGD(momentum=0.9) if Op == 'sgd' else tf.optimizers.Adam(),
        metrics=scoring
    )
    
    return model

def fit(
        model, 
        X_train : np.ndarray, 
        X_test  : np.ndarray, 
        y_train : np.ndarray, 
        y_test  : np.ndarray , 
        augment : bool = True, 
        epochs  : int = 40, 
        batch_size_train : int = 32,
        batch_size_test_or_val : int = 8,
        subset : str = 'test',
        use_callbacks : bool = False
        ):
    
    model = tf.keras.models.Model()

    if augment is True:
        datagen = data_augmenter_v2()
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        if use_callbacks:
            callback_tf, callbacks_models, callbacks_best_models = backup(epoch=epochs)

        # fits the model on batches with real-time data augmentation:
        if subset == 'validation' : 
            model.fit(datagen.flow(X_train, y_train, batch_size=batch_size_train,
                    subset='training'),
                    validation_data=datagen.flow(X_train, y_train,
                    batch_size=batch_size_test_or_val, subset=subset),
                    steps_per_epoch= int( len(X_train) / batch_size_train) , 
                    epochs=epochs,
                    callbacks=[ callback_tf, callbacks_models, callbacks_best_models] if use_callbacks else None
                    )

        elif subset == 'test':
            model.fit(datagen.flow(X_train, y_train, batch_size=batch_size_train,
                    subset='training'),
                    validation_data=datagen.flow(X_test, y_test,
                    batch_size=batch_size_test_or_val, subset=subset),
                    steps_per_epoch=int( len(X_train) / batch_size_train), 
                    epochs=epochs,
                    callbacks=[ callback_tf, callbacks_models, callbacks_best_models] if use_callbacks else None
                    )
        else: print("subset should be 'test' ot 'validation'")
    else:
        # do not use data augmentation 
        if subset == 'validation' : 
            model.fit(X_train, y_train,
                    batch_size=batch_size_train,
                    epochs=epochs,
                    validation_data=(X_train, y_train),
                    verbose=1,
                    callbacks=[ callback_tf, callbacks_models, callbacks_best_models] if use_callbacks else None
                )
   
        elif subset == 'test':
            model.fit(X_train, y_train,
                    batch_size=batch_size_train,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    verbose=1,
                    callbacks=[ callback_tf, callbacks_models, callbacks_best_models] if use_callbacks else None
                )
        else: print("subset should be 'test' ot 'validation'")

    return model

def evaluation(model, X, y):
    scoring = model.evaluate(x=X, y=y, verbose=1)
    return scoring 

def prediction(model, X, best_one : bool = True):
    import numpy as np 

    y_pred = model.predict(X)
    if best_one is True: y_pred  = np.argmax(y_pred, axis=1)
    else: pass 

    return y_pred 

def split_data(
            X : np.ndarray, 
            y : np.ndarray, 
            normalize       : bool = True, 
            random_state    : int = 1, 
            test_size       : float = 0.2, 
            classes         : int = 12
            ):
    from sklearn.model_selection import train_test_split as tts 

    X_train, XX, y_train, yy = tts(X, y, random_state=random_state, shuffle=True, test_size=test_size)
    X_dev, X_test, y_dev, y_test = tts(XX, yy, random_state=random_state, shuffle=True, test_size=0.5)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=classes)
    y_test  = tf.keras.utils.to_categorical(y=y_test, num_classes=classes)

    if normalize is True:
        X_train = X_train / 255. 
        X_test  = X_test / 255.
        X_dev   = X_dev  / 255.
    
    else : pass 

    data = {
        "X_train" : X_train, # train set
        "y_train" : y_train, # train set 
        'X_test'  : X_test , # test set
        'y_test'  : y_test , # test set
        'X_dev'   : X_dev  , # validation set
        "y_dev"   : y_dev    # validation set 
        }

    return data
    
def backup(epoch):

    log_dir = "./embeding_models/logs"
    callback_tf = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    path_models = "./embeding_models/model-{epoch:04d}.h5"
    callbacks_models = tf.keras.callbacks.ModelCheckpoint(filepath=path_models, verbose=1)
    
    path_best_models = "./embeding_models/best-model.h5"
    callbacks_best_models = tf.keras.callbacks.ModelCheckpoint(filepath=path_best_models, 
                            monitor="val_accuracy", verbose=1, save_best_only=True)
    
    return callback_tf, callbacks_models, callbacks_best_models

def create_json_file(report):
    import json, os

    try:
        json_path = f"./embeding/json_path_{1}//output.json"
        report['description'] = ["description"]

        with open(json_path, "w") as file:
            json.dump(report, file, indent=4)
    except FileNotFoundError:
        json_path =  f"./embeding//json_path_{1}"
        os.makedirs(json_path)
        json_path = f"./embeding/json_path_{1}/output.json"

        with open(json_path, "w") as file:
            json.dump(report, file, indent=4)

    print('\nReport saved as ', json_path)