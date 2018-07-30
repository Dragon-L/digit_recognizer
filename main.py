import pandas as pd
import tensorflow as tf
import numpy as np


def get_data():
    train_data = pd.read_csv('./data/train.csv').values
    test_data = pd.read_csv('./data/test.csv').values

    eval_x, eval_y, train_x, train_y = split_train_set(train_data, test_size=0.1)
    test_x = test_data.reshape(-1, 28, 28, 1)

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     rotation_range=10,
    #     zoom_range=0.1,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1
    # )
    # datagen.fit()

    return train_x/255.0, train_y, eval_x/255.0, eval_y, test_x/255.0


def split_train_set(train_data, test_size):
    np.random.shuffle(train_data)
    index = int((1 - test_size) * train_data.shape[0])
    train_x = train_data[:index, 1:].reshape(-1, 28, 28, 1)
    train_y = train_data[:index, 0].reshape(-1, 1)
    eval_x = train_data[index:, 1:].reshape(-1, 28, 28, 1)
    eval_y = train_data[index:, 0].reshape(-1, 1)
    return eval_x, eval_y, train_x, train_y


def cnn_model(features, labels, mode):
    features = tf.to_float(features)
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding='same',activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.reshape(pool3, [-1, 3 * 3 * 128])
    dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        'number': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits=logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss =tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metris_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['number'])
        }
        return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=eval_metris_ops, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def main(arg):
    tf.logging.set_verbosity(tf.logging.INFO)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model,
        model_dir='./output',
    )

    train_x, train_y, eval_x, eval_y, test_x = get_data()

    # train_log = {'probabilities': 'softmax_tensor'}
    # logging_hook = tf.train.LoggingTensorHook(tensors=train_log, every_n_iter=20)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_x, y=train_y, batch_size=128, num_epochs=1000, shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=200, max_steps=10000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_x, y=eval_y, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_x, shuffle=False)
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)
    predictions_array = []
    for pred in predictions:
        predictions_array.append(pred['number'])
    submission = pd.DataFrame(data=predictions_array, index=np.arange(1, test_x.shape[0] + 1))
    submission.to_csv('./data/submission', header=['Label'], index_label='ImageId')

if __name__ == '__main__':
    tf.app.run()
