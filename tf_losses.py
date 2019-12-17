# coding=utf-8
# Author: Didia
# Date: 19-12-17
import tensorflow as tf
import keras.backend as K
from keras import losses


def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def focal_loss_binary(y_true, y_pred):
    alpha, gamma = 2.0, 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def focal_loss(y_true, y_pred, ignore_value=255):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """
    print(y_true.shape, y_pred.shape)
    if ignore_value:
        raw_prediction = tf.reshape(y_pred, [-1, 5])
        raw_gt = tf.reshape(y_true, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, 6 - 1)), 1) # eliminate those pixels with ignore value
        y_true = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        y_pred = tf.gather(raw_prediction, indices)
    print(y_true.shape, y_pred.shape)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    loss = tf.reduce_mean(loss)
    return loss


def com_con(x, y):
    return x + y


def multi_weighted_dense_dice_loss(augloss=True, OC=3):
    def loss(y_true, y_pred):

        pred_list = tf.unstack(y_pred, num=OC, axis=-1)
        true_list = tf.unstack(y_true, num=OC, axis=-1)
        dice_loss = 0

        for i in range(len(pred_list)):
            pred = pred_list[i]
            true = true_list[i]

            dice_loss += dice_coef_loss(true, pred)
        com_loss = 0
        a = 0
        for i in range(OC - 1):
            for j in range(i + 1, OC):
                dice2 = dice_coef_loss(com_con(true_list[i], true_list[j]),
                                       com_con(pred_list[i], pred_list[j]))
                com_loss = com_loss + dice2 * (1 / (j - i))
                a = a + (1 / (j - i))
        a = OC / a
        b = 1
        if not augloss:
            a = 0
        return b * dice_loss + a * com_loss

    return loss


def dice(c=0, OC=3):
    def Dice(y_true, y_pred):
        pred = tf.unstack(y_pred, num=OC, axis=3)[c]
        true = tf.unstack(y_true, num=OC, axis=3)[c]

        dice = dice_coef(true, pred)
        return dice

    return Dice


def mean_dice(OC=3):
    def Dice(y_true, y_pred):
        dice = 0
        for i in range(OC):
            pred = tf.unstack(y_pred, num=OC, axis=3)[i]
            true = tf.unstack(y_true, num=OC, axis=3)[i]
            dice += dice_coef(true, pred)
        return dice / OC

    return Dice


def multi_dice_com_value(OC):
    def loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        pred = tf.unstack(y_pred, num=OC, axis=3)
        true = tf.unstack(y_true, num=OC, axis=3)
        com_loss = 0

        for i in range(OC):
            for j in range(i + 1, OC):
                dice2 = dice_coef_loss(com_con(true[i], true[j]),
                                       com_con(pred[i], pred[j]))
                com_loss = com_loss + dice2 * (1 / (j - i))

        return com_loss
    return loss


def dice_coef_3d(y_true, y_pred, epsilon=1e-5):
    dice_numerator = 2.0 * K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
    dice_denominator = K.sum(K.square(y_true), axis=[1, 2, 3, 4]) + K.sum(K.square(y_pred), axis=[1, 2, 3, 4])

    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return K.mean(dice_score, axis=0)


def dice_coef_loss_3d(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)