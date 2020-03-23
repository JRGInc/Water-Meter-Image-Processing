__author__ = 'Larry A. Hartman'
__company__ = 'Janus Research'

import csv
import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
import time as ttime
from common import data_ops, errors, file_ops
from config.core import CoreCfg
from datetime import *
from machine import inception_v4  # , mobilenet_v3_large
from tensorflow import keras


def version():
    """
    Returns Tensor Flow version
    """
    print(tf.__version__)


def tf_init():
    """
    Clears Tensor Flow session
    """
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    np.set_printoptions(
        threshold=np.inf,
        precision=None
    )


def train(
    tf_dict: dict
):
    """
    Builds Tensor Flow model

    :param tf_dict: dict
    """
    logfile = 'januswm-train'
    logger = logging.getLogger(logfile)

    timea = ttime.time()

    core_cfg = CoreCfg()
    core_path_dict = core_cfg.get('core_path_dict')
    img_path_dict = core_cfg.get('img_path_dict')

    learn_err = False

    try:
        tf_init()
        train_ds, train_count = data_ops.prepare_data(
            mode='train',
            tf_dict=tf_dict,
            img_root_dir=img_path_dict['trn'],
            img_cache_dir=core_path_dict['train_cache']
        )
        valid_ds, valid_count = data_ops.prepare_data(
            mode='train',
            tf_dict=tf_dict,
            img_root_dir=img_path_dict['vld'],
            img_cache_dir=core_path_dict['valid_cache']
        )
        train_steps = np.ceil(train_count / tf_dict['batch_size'])
        valid_steps = np.ceil(train_count / tf_dict['batch_size'])
        log = 'Train image count: {0}, steps: {1}: '.format(train_count, train_steps)
        logger.info(log)
        print(log)
        log = 'Validation image count: {0}, steps: {1}: '.format(valid_count, valid_steps)
        logger.info(log)
        print(log)

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=tf_dict['patience'],
                monitor='val_loss',
                min_delta=1e-4,
                mode='auto',
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    core_path_dict['mdl'],
                    'keras/tf_mdl_{epoch:05d}-{val_loss:7.5f}.h5'
                ),
                save_best_only=True,
                save_freq='epoch',
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    core_path_dict['wgts'],
                    'periodic/tf_cp_{epoch:05d}-{val_loss:7.5f}'
                ),
                save_best_only=True,
                save_weights_only=True,
                save_freq='epoch',
                monitor='val_loss',
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=core_path_dict['logs'],
                update_freq='epoch',
                histogram_freq=1
            )
        ]

        model = inception_v4.create_inception_v4(
            tf_dict=tf_dict
        )
        # model = mobilenet_v3_large.MobileNetV3Large(tf_dict=tf_dict)
        # model.build(input_shape=(
        #     tf_dict['batch_size'],
        #     tf_dict['img_tgt_width'],
        #     tf_dict['img_tgt_height'],
        #     tf_dict['nbr_channels']
        # ))

        # Specify the training configuration (optimizer, loss, metrics)
        learn_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=0.001,
            decay_steps=train_steps * 5,
            decay_rate=1,
            staircase=False
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learn_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
        )
        model.summary()
        mdl_hist = model.fit(
            x=train_ds,
            epochs=tf_dict['epochs'],
            steps_per_epoch=train_steps,
            validation_data=valid_ds,
            validation_steps=valid_steps,
            callbacks=callbacks,
            verbose=1
        )

        model.save_weights(
            filepath=os.path.join(
                core_path_dict['wgts'],
                'final/final.ckpt'
            ),
            overwrite=True,
            save_format='h5'
        )

        nbr_epochs = len(mdl_hist.history['val_loss'])
        log = 'Training ended at {0} epochs with {1} seconds elapsed.'.\
            format(nbr_epochs, round((ttime.time() - timea), 3))
        logger.info(log)
        epochs_arr = np.arange(0, nbr_epochs)
        title = 'Training Loss and Accuracy'

        # plot the training loss and accuracy
        plt.style.use(style='ggplot')
        plt.figure()
        plt.plot(
            epochs_arr,
            mdl_hist.history['loss'],
            label='train_loss'
        )
        plt.plot(
            epochs_arr,
            mdl_hist.history['val_loss'],
            label='val_loss'
        )
        plt.plot(
            epochs_arr,
            mdl_hist.history['sparse_categorical_accuracy'],
            label='train_acc'
        )
        plt.plot(
            epochs_arr,
            mdl_hist.history['val_sparse_categorical_accuracy'],
            label='val_acc'
        )
        plt.title(label=title)
        plt.xlabel(xlabel='Epoch #')
        plt.ylabel(ylabel='Loss/Accuracy')
        plt.legend()
        plt.savefig(os.path.join(
            core_path_dict['wgts'], 'loss and accuracy.png'
        ))

    except Exception as exc:
        learn_err = True
        log = exc
        logger.error(log)
        print(log)

    print('Train model time elapsed: {0} sec'.format(ttime.time() - timea))
    return learn_err


def test(
    tf_dict: dict,
    value: str
):
    """
    Tests Tensor Flow model

    :param tf_dict: dict
    :param value: str
    """
    logfile = 'januswm-test'
    logger = logging.getLogger(logfile)

    timea = ttime.time()

    core_cfg = CoreCfg()
    train_version = core_cfg.get(attrib='train_version')
    test_version = core_cfg.get(attrib='test_version')
    test_set = core_cfg.get(attrib='test_set')
    core_path_dict = core_cfg.get(attrib='core_path_dict')
    data_path_dict = core_cfg.get(attrib='data_path_dict')
    img_path_dict = core_cfg.get(attrib='img_path_dict')
    test_url_str = os.path.join(
        data_path_dict['rslt'],
        'Predictions_v' + value + '_' + train_version + '_' + test_version + '_' + test_set + '_' +
        datetime.today().strftime('%Y-%m-%d_%H%M') + '.csv'
    )

    test_err = False
    correct = 0
    error = 0
    index = 0
    label_val = None
    prediction = None
    probability = None

    try:
        with open(test_url_str, mode='a') as test_file:
            test_file_writer = csv.writer(
                test_file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            test_file_writer.writerow([
                'Sample',
                'Label',
                'Prediction',
                'Probability',
                'Error',
                'Gross',
                'Major Transition',
                'Minor Transition',
                'Samples',
                'Errors',
                'Samples',
                'Errors',
                'Samples',
                'Errors',
                'Samples',
                'Errors',
                'Samples',
                'Errors',
                'Samples',
                'Errors'
            ])

        version()
        tf_init()
        model_test = inception_v4.create_inception_v4(
            tf_dict=tf_dict
        )
        model_test.load_weights(
            filepath=os.path.join(core_path_dict['wgts'], 'final/final.ckpt'),
            by_name=False
        )
        # mdl_url_str = '/opt/Janus/WM/model/pb/saved_model.pb'
        # model_test = tf.keras.models.load_model(mdl_url_str)
        model_test.summary()

        img_path = os.path.join(img_path_dict['test'], value)
        for root, dirs, images in os.walk(top=img_path):
            for img_name in images:

                d0_err = 0
                d1_err = 0
                d2_err = 0
                d3_err = 0
                d4_err = 0
                d5_err = 0

                d0_sam = 0
                d1_sam = 0
                d2_sam = 0
                d3_sam = 0
                d4_sam = 0
                d5_sam = 0

                gross_err = 0
                maj_trans_err = 0
                min_trans_err = 0

                digit = img_name.split(sep='_')[0]

                if digit == 'd0':
                    d0_sam = 1
                elif digit == 'd1':
                    d1_sam = 1
                elif digit == 'd2':
                    d2_sam = 1
                elif digit == 'd3':
                    d3_sam = 1
                elif digit == 'd4':
                    d4_sam = 1
                elif digit == 'd5':
                    d5_sam = 1

                image_url = os.path.join(root, img_name)
                test_ds, test_count = data_ops.prepare_data(
                    mode='test',
                    img_url=image_url,
                    tf_dict=tf_dict
                )
                print(test_ds)
                test_steps = np.ceil(test_count / tf_dict['batch_size'])

                log = 'Image: {0}'.format(img_name)
                logger.info(log)
                print(log)

                for record in test_ds:
                    for label in record[1]:
                        label_val = int(label.numpy())

                predictions = model_test.predict(
                    x=test_ds,
                    steps=test_steps
                )
                for element in predictions:
                    prediction = element.argmax(axis=0)
                    print(prediction)
                    probability = element[prediction]
                    print(probability)

                if label_val == prediction:
                    correct += 1
                    incorrect = 0

                else:
                    error += 1
                    incorrect = 1

                    if digit == 'd0':
                        d0_err = 1
                    elif digit == 'd1':
                        d1_err = 1
                    elif digit == 'd2':
                        d2_err = 1
                    elif digit == 'd3':
                        d3_err = 1
                    elif digit == 'd4':
                        d4_err = 1
                    elif digit == 'd5':
                        d5_err = 1

                    if label_val < 10:
                        if (prediction == (label_val + 10)) or (prediction == (label_val - 10)):
                            min_trans_err = 1
                        elif (prediction == (label_val + 1)) or (prediction == (label_val - 1)):
                            maj_trans_err = 1
                        else:
                            gross_err = 1

                    elif (label_val >= 10) and (label_val < 20):
                        if (prediction == (label_val - 9)) or (prediction == (label_val - 10)):
                            min_trans_err = 1
                        else:
                            gross_err = 1

                    else:
                        gross_err = 1

                    img_error_url = os.path.join(
                        img_path_dict['test_err'],
                        str(label_val),
                        str(prediction),
                        img_name
                    )

                    shutil.copy2(
                        src=image_url,
                        dst=img_error_url
                    )

                    log = 'Incorrect prediction: label {0}, prediction {1}'.format(label_val, prediction)
                    logger.warning(log)
                    print(log)

                with open(test_url_str, mode='a') as test_file:
                    test_file_writer = csv.writer(
                        test_file,
                        delimiter=',',
                        quotechar='"',
                        quoting=csv.QUOTE_MINIMAL
                    )
                    test_file_writer.writerow([
                        img_name,
                        label_val,
                        prediction,
                        probability,
                        incorrect,
                        gross_err,
                        maj_trans_err,
                        min_trans_err,
                        d5_sam,
                        d5_err,
                        d4_sam,
                        d4_err,
                        d3_sam,
                        d3_err,
                        d2_sam,
                        d2_err,
                        d1_sam,
                        d1_err,
                        d0_sam,
                        d0_err
                    ])

                index += 1

                print('Interim test results, image count: {0}, error count: {1}, accuracy: {2}% \n\n'.
                      format(index, error, round(float(correct / index), 4) * 100))

    except Exception as exc:
        test_err = True
        log = exc
        logger.error(log)
        print(log)

    if index > 0:
        log = 'Final test results, image count: {0}, error count: {1}, accuracy: {2}%'.\
            format(index, error, round(float(correct / index), 4) * 100)
        logger.info(log)
        print(log)

    else:
        log = 'Final test results, no images to test.'
        logger.info(log)
        print(log)

    print('Model test time elapsed: {0} sec'.format(ttime.time() - timea))
    return test_err


def predict(
    err_xmit_url: str,
    img_orig_dtg: str,
    img_digw_url: str,
    img_seq: str,
    tf_dict: dict
) -> [bool, list]:
    """
    Generates and returns predictions for each digit using TensorFlow

    :param err_xmit_url: str
    :param img_orig_dtg: str
    :param img_digw_url: str
    :param img_seq: str
    :param tf_dict: dict

    :return pred_err: bool
    :return pred_list: list
    """
    logfile = 'januswm-test'
    logger = logging.getLogger(logfile)

    timea = ttime.time()

    pred_err = False
    info = inspect.getframeinfo(frame=inspect.stack()[1][0])
    err_msg_base = 'FILE: ' + info.filename + ' ' + 'FUNCTION: ' + info.function

    core_cfg = CoreCfg()
    data_path_dict = core_cfg.get('data_path_dict')
    cfg_url_dict = core_cfg.get('cfg_url_dict')
    img_path_dict = core_cfg.get('img_path_dict')

    predictions = [
        [0, None],
        [0, None],
        [0, None],
        [0, None],
        [0, None],
        [0, None],
    ]
    pred_list = [
        img_orig_dtg,
        '',
        'V'
    ]

    try:
        mdl_url_str = '/opt/Janus/WM/model/tflite/mdl_{0}.tflite'.format(core_cfg.get(attrib='train_version'))
        interpreter = tf.lite.Interpreter(model_path=mdl_url_str)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for digit in range(5, -1, -1):
            timeb = ttime.time()
            img_pdig_url = os.path.join(
                img_path_dict['pred'],
                'pred' + '_d' + str(5 - digit) + os.path.basename(img_digw_url)[4::]
            )
            if os.path.isfile(path=img_pdig_url):
                data_err, img_pdig = data_ops.image_to_pred_data(
                    img_pdig_url=img_pdig_url,
                    tf_dict=tf_dict
                )
                # print(img_pdig)
                # print(img_pdig.shape)

                if not data_err:
                    interpreter.set_tensor(input_details[0]['index'], img_pdig)
                    interpreter.invoke()
                    pred_outputs = interpreter.get_tensor(output_details[0]['index'])

                    for element in pred_outputs:
                        # print(index)
                        predictions[5 - digit][0] = element.argmax(axis=0)
                        predictions[5 - digit][1] = element[predictions[digit][0]]
                        print('Prediction is {0}, with probability {1}'.
                              format(predictions[digit][0], predictions[digit][1]))

                    if predictions[5 - digit][0] >= 10:
                        pred_list[2] = 'I'
                    else:
                        pred_list[1] = str(predictions[5 - digit][0]) + pred_list[1]

                    if predictions[5 - digit][0] == 10:
                        pred_list[1] += 'A'

                        log = 'Prediction value for digit {0} is between 0 and 1.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 11:
                        pred_list[1] += 'B'

                        log = 'Prediction value for digit {0} is between 1 and 2.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 12:
                        pred_list[1] += 'C'

                        log = 'Prediction value for digit {0} is between 2 and 3.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 13:
                        pred_list[1] += 'D'

                        log = 'Prediction value for digit {0} is between 3 and 4.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 14:
                        pred_list[1] += 'E'

                        log = 'Prediction value for digit {0} is between 4 and 5.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 15:
                        pred_list[1] += 'F'

                        log = 'Prediction value for digit {0} is between 5 and 6.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 16:
                        pred_list[1] += 'G'

                        log = 'Prediction value for digit {0} is between 6 and 7.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 17:
                        pred_list[1] += 'H'

                        log = 'Prediction value for digit {0} is between 7 and 8.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 18:
                        pred_list[1] += 'I'

                        log = 'Prediction value for digit {0} is between 8 and 9.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 19:
                        pred_list[1] += 'J'

                        log = 'Prediction value for digit {0} is between 9 and 0.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 20:
                        pred_list[1] += 'K'

                        log = 'Prediction value for digit {0} is occluded over 0.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 21:
                        pred_list[1] += 'L'

                        log = 'Prediction value for digit {0} is occluded over 1.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 22:
                        pred_list[1] += 'M'

                        log = 'Prediction value for digit {0} is occluded over 2.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 23:
                        pred_list[1] += 'N'

                        log = 'Prediction value for digit {0} is occluded over 3.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 24:
                        pred_list[1] += 'O'

                        log = 'Prediction value for digit {0} is occluded over 4.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 25:
                        pred_list[1] += 'P'

                        log = 'Prediction value for digit {0} is occluded over 5.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 26:
                        pred_list[1] += 'Q'

                        log = 'Prediction value for digit {0} is occluded over 6.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 27:
                        pred_list[1] += 'R'

                        log = 'Prediction value for digit {0} is occluded over 7.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 28:
                        pred_list[1] += 'S'

                        log = 'Prediction value for digit {0} is occluded over 8.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    elif predictions[5 - digit][0] == 29:
                        pred_list[1] += 'T'

                        log = 'Prediction value for digit {0} is occluded over 9.'.format(digit)
                        logger.info(msg=log)
                        print(log)

                    pred_list.append(predictions[5 - digit][0])
                    pred_list.append(predictions[5 - digit][1])

                else:
                    pred_err = True
                    log = 'Error loading data from {0}, could not make prediction.'.format(img_pdig_url)
                    logger.error(msg=log)
                    print(log)

                    pred_list[1] += 'Z'
                    pred_list[2] = 'I'

                    err_msg = err_msg_base + ' ' + \
                        'MESSAGE: ' + log + '\n'
                    errors.errors(
                        err_xmit_url=err_xmit_url,
                        err_msg=err_msg
                    )

            else:
                pred_err = True
                log = 'Could not locate file {0} from which to make predictions.'.format(img_pdig_url)
                logger.error(msg=log)
                print(log)

                pred_list[1] += 'Z'
                pred_list[2] = 'I'

                err_msg = err_msg_base + ' ' + \
                    'MESSAGE: ' + log + '\n'
                errors.errors(
                    err_xmit_url=err_xmit_url,
                    err_msg=err_msg
                )

            print('Digit prediction time elapsed: {0} sec'.format(ttime.time() - timeb))

        if pred_list[2] == 'V':
            log = 'Prediction value has valid digits.'
            logger.info(msg=log)
            print(log)

        else:
            log = 'Prediction value has one or more invalid digits.'
            logger.info(msg=log)
            print(log)

        # Save prediction
        last_url_str = os.path.join(
            data_path_dict['last'],
            'last_' + img_orig_dtg + '_' + img_seq + '.txt'
        )
        copy_err = file_ops.copy_file(
            data_orig_url=cfg_url_dict['last'],
            data_dest_url=last_url_str
        )
        if not copy_err:
            file_ops.f_request(
                file_cmd='file_csv_writelist',
                file_name=last_url_str,
                data_file_in=pred_list
            )

        hist_url_str = os.path.join(
            data_path_dict['hist'],
            'hist_' + img_orig_dtg.split('_')[0] + '.txt'
        )
        if not os.path.isfile(path=hist_url_str):
            copy_err = file_ops.copy_file(
                data_orig_url=cfg_url_dict['hist'],
                data_dest_url=hist_url_str
            )
            if not copy_err:
                open(hist_url_str, 'w').close()

        # Dump every prediction into file for history
        file_ops.f_request(
            file_cmd='file_csv_appendlist',
            file_name=hist_url_str,
            data_file_in=pred_list
        )

        log = 'Successfully predicted digit values from numpy array.'
        logger.info(msg=log)

        # img_ops.remove_images(img_path=img_path_dict['pred'])
        # log = 'Deleted prediction images from disk.'
        # logger.info(msg=log)
        # print(log)

    except Exception as exc:
        pred_err = True
        log = 'Failed to predict digit values from image data.'
        logger.error(msg=log)
        logger.error(msg=exc)
        print(log)
        print(exc)

        err_msg = err_msg_base + ' ' + \
            'MESSAGE: ' + log + '\n'
        errors.errors(
            err_xmit_url=err_xmit_url,
            err_msg=err_msg
        )

    if not pred_err:
        img_olay_text = '          Value: '
        for digit in range(5, -1, -1):
            if digit > 0:
                img_olay_text += str(pred_list[1][5 - digit]) + '-'
            else:
                img_olay_text += str(pred_list[1][5 - digit])

        if pred_list[2] == 'V':
            img_olay_text += '     (valid)'
        elif pred_list[2] == 'I':
            img_olay_text += '     (invalid)'

    else:
        img_olay_text = '           Value: Prediction Error'

    print('Total prediction time elapsed: {0} sec'.format(ttime.time() - timea))

    return pred_err, pred_list, img_olay_text
