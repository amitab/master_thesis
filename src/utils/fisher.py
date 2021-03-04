from keras.models import Model
import tensorflow as tf
import numpy as np

def get_fisher_info(model, raw_data, label_data):
    y = model.output

    row_idx = tf.range(tf.shape(y)[0])
    col_idx = tf.argmax(y, axis=1, output_type=tf.dtypes.int32)
    full_indices = tf.stack([row_idx, col_idx], axis=1)
    fx_tensors = tf.gather_nd(y, full_indices)

    x_tensors = model.trainable_weights

    num_samples = 100
    m = Model(inputs=model.input, outputs=fx_tensors)

    fisher_information = []
    for v in range(len(x_tensors)):
        fisher_information.append(np.zeros(x_tensors[v].get_shape().as_list()).astype(np.float32))

    for i in range(num_samples):
        data_idx = np.random.randint(raw_data.shape[0])
        sampled_data = raw_data[data_idx:data_idx+1]
        sampled_input_variables = [ sampled_data ]
        print ('sample num: %4d, data_idx: %5d' % (i, data_idx))

        with tf.GradientTape() as tape:
            p = m(sampled_data)
            lo = tf.math.log(p)

        gradients = tape.gradient(lo, x_tensors)
        derivatives = [g.numpy() for g in gradients]
        prob = p.numpy()[0]
        
    #     derivatives, prob = sess.run([tf.gradients(tf.log(fx_tensors), x_tensors), fx_tensors],
    #     feed_dict={t: v for t,v in zip(input_tensors, sampled_input_variables)})

        for v in range(len(fisher_information)):
            fisher_information[v] += np.square(derivatives[v]) * prob
        
    return fisher_information