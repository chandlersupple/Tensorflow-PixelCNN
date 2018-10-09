# Libraries
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

sess = tf.InteractiveSession()

# Operations

    # mask
def mask(mask_type, num_inputs, num_outputs, kernel_size, num_channels):

    ones_mask = np.ones([kernel_size[0], kernel_size[1], num_inputs, num_outputs])
    kernel_ch = kernel_size[0] // 2
    kernel_cw = kernel_size[1] // 2
    
    if mask_type == 'v':
        ones_mask[kernel_ch:, :, :, :] = 0
    else:
        ones_mask[kernel_ch, kernel_cw + 1:, :, :] = 0
        ones_mask[kernel_ch + 1:, :, :, :] = 0
        
        if mask_type == 'b':
            m_pixel = lambda i,j: i > j
        if mask_type == 'a':
            m_pixel = lambda i,j: i >= j
        
        for i in range(num_inputs):
            for j in range(num_outputs):
                if m_pixel(i % num_channels, j % num_channels):
                    ones_mask[kernel_ch, kernel_cw, i, j] = 0
                    
    return ones_mask
        
    # conv2d
def conv2d(x, mask_type, num_outputs, kernel_size, strides, num_channels, scope= "conv2d",
           activation_func= None, padding= "SAME"):
    with tf.variable_scope(scope):
        
        if mask_type == 'v' and kernel_size == [1, 1]:
            mask_type = None
        
        num_inputs = np.shape(x)[-1]
        kernel_h, kernel_w = kernel_size[0], kernel_size[1]
        w_init = tf.contrib.layers.xavier_initializer()
        weights = tf.get_variable("weights", [kernel_h, kernel_w, num_inputs, num_outputs], tf.float32, w_init)
        
        if mask_type is not None:
            w_mask = mask(mask_type, num_inputs, num_outputs, kernel_size, num_channels)
            weights = weights * tf.constant(w_mask, dtype= tf.float32)
            
        output = tf.nn.conv2d(x, weights, strides, padding)
        b_init = tf.zeros_initializer()
        biases = tf.get_variable("biases", [num_outputs,], tf.float32, b_init)
        output = tf.nn.bias_add(output, biases)
        
        if activation_func:
            output = activation_func(output)
            
        return output
    
    # G_conv
def G_conv(x, kernel_size, num_channels, G_conv_fm, scope= "G_conv"):
    with tf.variable_scope(scope):
        
        h_x, v_x = tf.split(x, 2, 3)
        two_p = G_conv_fm
        p = two_p // 2
        
        v_nxn = conv2d(x, 'v', two_p, kernel_size, [1, 1, 1, 1], num_channels, scope= "v_nxn")
        v_out = G_activation_func(v_nxn, 'v', kernel_size, num_channels, scope= "v_out")
        
        v_oxo = conv2d(v_nxn, 'v', two_p, [1, 1], [1, 1, 1, 1], num_channels, scope= "v_oxo")
        h_oxn = conv2d(h_x, 'b', two_p, [1, kernel_size[0]], [1, 1, 1, 1], num_channels, scope= "h_oxn")
        h_in = h_oxn + v_oxo
    
        h_out = G_activation_func(h_in, 'b', kernel_size, num_channels, scope= "h_out")
        h_oxo = conv2d(h_out, 'b', p, [1, 1], [1, 1, 1, 1], num_channels, scope= "h_oxo")
        h_outputs = h_oxo + h_x
        
        return tf.concat([h_outputs, v_out], 3)
    
    # G_activation_func
def G_activation_func(x, mask_type, kernel_size, num_channels, scope= "G_activation_func"):
    with tf.variable_scope(scope):
        
        two_p = np.shape(x)[-1]
        d_conv = conv2d(x, mask_type, two_p, kernel_size, [1, 1, 1, 1], num_channels, scope= 'd_conv')
        d_in_one, d_in_two = tf.split(d_conv, 2, 3)
        d_tanh = tf.nn.tanh(d_in_one)
        d_sigm = tf.nn.sigmoid(d_in_two)
    
        return d_tanh * d_sigm

    # batch
def batch(batches_in_epoch, batch_size, image_dims, flat= False): # Returns a batch of images from the given dataset
    
        all_batches = []
        for batch_iter in range (batches_in_epoch):
            batch_iarr = []
            for inst in range (batch_iter * batch_size + 1, batch_iter * batch_size + batch_size + 1):
                image_inst = image.load_img(('Img-%s.jpg' %(inst)), target_size = [image_dims[0], image_dims[1]])
                inst_arr = image.img_to_array(image_inst)
                if flat == True:
                    inst_arr = np.reshape(inst_arr, [(image_dims[0])**2])
                batch_iarr.append(inst_arr)
            print('Batch %s / %s' %(batch_iter, batches_in_epoch))
            all_batches.append(batch_iarr)
        return all_batches

# Network

class PixelCNN():
    def __init__(self, image_height, image_width, num_channels, q_levels= 256, G_conv_fm= 16, G_conv_layers= 7, 
                 O_conv_fm= 32, lr= 0.0001, grad_clip= 1):
        
        # Training is done in parallel, sampling is sequential
        
        self.img_h = image_height
        self.img_w = image_width
        self.num_channels = num_channels
        self.pixel_d = 256
        self.q_levels = q_levels
        
        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, self.num_channels])
        self.y = tf.placeholder(tf.int32, [None, self.img_h, self.img_w, self.num_channels])
                    
        in_conv = conv2d(self.x, 'a', G_conv_fm, [7, 7], [1, 1, 1, 1], self.num_channels, scope= "in_conv")
        
        mid_conv = G_conv(in_conv, [7, 7], self.num_channels, G_conv_fm, scope= "mid_conv0")

        for layer in range (1, G_conv_layers):
            scope = ("mid_conv%s" % layer)
            mid_conv = G_conv(mid_conv, [7, 7], self.num_channels, G_conv_fm, scope= scope)
        
        out_conv = tf.nn.relu(conv2d(mid_conv, 'b', O_conv_fm, [1, 1], [1, 1, 1, 1], self.num_channels, 
                                     scope= "out_conv0"))
        
        self.logits = conv2d(out_conv, 'b', self.q_levels * self.num_channels, [1, 1], [1, 1, 1, 1], self.num_channels,
                             scope= "out_conv1")
                
        if self.num_channels > 1:
            self.logits = tf.reshape(self.logits, [-1, self.img_h, self.img_w, self.q_levels, self.num_channels])
            self.logits = tf.transpose(self.logits, perm= [0, 1, 2, 4, 3])
            
        flat_logits = tf.reshape(self.logits, [-1, q_levels])
        y_flat = tf.reshape(self.y, [-1])
        
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels= y_flat, logits= flat_logits))
    
        flat_output = tf.nn.softmax(flat_logits)
        self.output = tf.reshape(flat_output, [-1, self.img_h, self.img_w, self.num_channels, self.q_levels])
        
        optimizer = tf.train.RMSPropOptimizer(lr)
        t_gv = optimizer.compute_gradients(self.loss)
        td_gv = [(tf.clip_by_value(gv[0], -grad_clip, grad_clip), gv[1]) for gv in t_gv]
        self.train_op = optimizer.apply_gradients(td_gv)

        sess.run(tf.global_variables_initializer())        
        
    def train(self, images):
        
        _, cost = sess.run([self.train_op, self.loss], 
                           feed_dict= {self.x: images, self.y: images})
        return cost
        
    def predict(self, images):
        
        pixel_probs = sess.run(self.output, feed_dict= {self.x: images})
        pixel_indices = np.argmax(pixel_probs, 4)
        pixel_values = np.multiply(pixel_indices, (self.pixel_d - 1) / (self.q_levels - 1))
        
        return pixel_values
        
    def sample(self, images, occlusion):
        
        b_occlusion = occlusion
        images_seq = images
        images_seq[:, b_occlusion:, :, :] = 0
        
        for i in range (b_occlusion, self.img_h):
            for j in range (self.img_w):
                for k in range (self.num_channels):
                    n_sample = self.predict(images_seq)
                    images_seq[:, i, j, k] = n_sample[:, i, j, k]
                    
        return images_seq
                    
batch_size = 100
batches_in_epoch = 13000 // batch_size
network = PixelCNN(32, 32, 3)
batch_xy = batch(batches_in_epoch, batch_size, [32, 32])

for epoch_iter in range (128):
    for batch_iter in range (batches_in_epoch):
        cost = network.train(batch_xy[batch_iter])
        print('Epoch %s, Batch %s / %s, Cost: %s' %(epoch_iter, batch_iter, batches_in_epoch, int(cost * 10e4))) 

batch_size = 1
resh_batch = np.reshape(batch_xy[2][0], [batch_size, 32, 32, 3])
pred = network.sample(resh_batch)
resh_pred = np.reshape(pred, [32, 32, 3])
pred_img = image.array_to_img(resh_pred)
