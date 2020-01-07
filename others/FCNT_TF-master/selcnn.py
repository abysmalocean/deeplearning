

import tensorflow as tf
import numpy as np

from utils import variable_on_cpu, variable_with_weight_decay

class SelCNN:
	def __init__(self, scope, vgg_conv_layer):
		"""
		selCNN network class. Initialize graph.

		Args: 
			name: string, name of the network.
			vgg_conv_layer: tensor, either conv4_3 or cnv5_3 layer
				of a pretrained vgg16 network
		"""
		# Initialize network
		#初始化网络
		self.scope = scope
		self.input_layer = vgg_conv_layer

		# Unpack vgg_conv_layer for partial derivatives
		self.feature_maps = [vgg_conv_layer[...,i] for i in range(512)]
		self.variables = []
		self.params = {
		'dropout_rate': 0.3,
		'k_size': [3, 3, 512, 1],
		'wd': 0.5,
		'lr_initial': 1e-9, # -9是训练sel-CNN时候的学习率，-7是GNET,SNET的学习率
		'lr_decay_steps': 0,
		'lr_decay_rate':  1
		}
		with tf.name_scope(scope) as scope:
			self.pre_M = self._get_pre_M()
		self.pre_M_size = self.pre_M.get_shape().as_list()[1:3]
		


	def _get_pre_M(self):  #返回热力图
		"""Build the sel-CNN graph and returns predicted Heat map."""
		input_maps = tf.pack(self.feature_maps, axis=-1)
		dropout_layer = tf.nn.dropout(input_maps, self.params['dropout_rate'])

		
		#给卷积层添加偏置
		kernel = variable_with_weight_decay(self.scope, 'kernel',\
							self.params['k_size'], wd = self.params['wd'])
		conv = tf.nn.conv2d(dropout_layer, kernel, [1,1,1,1], 'SAME')
		bias = variable_on_cpu(self.scope,'biases', [1], tf.constant_initializer(0.1))
		pre_M = tf.nn.bias_add(conv, bias)
		self.variables += [kernel, bias]
		# 减去均值
		#pre_M -= tf.reduce_mean(pre_M)
		#pre_M# /= tf.reduce_max(pre_M)
		return pre_M


	def train_op(self, gt_M_sz, global_step, add_regulizer=True):#利用第一帧训练网络#
		""" Train the network on the fist frame. 

		Args:
			gt_M_sz: tuple, shape identical to self.pre_M,
				Ground truth heatmap.
			add_regulizer: bool, True for adding L2 regulizer of the 
				kernel variables of the conv layer.

		Returns:
			train_op:
			total_losses:
			lr:
		"""
		
		self.gt_M = tf.placeholder(tf.float32, shape=gt_M_sz)
		pre_shape = self.pre_M.get_shape().as_list()[1:]

		assert gt_M_sz[1:] == pre_shape, \
			'Shapes are not compatiable! gt_M : {0}, pre_M : {1}'.format(
				gt_shape, pre_shape)
		
		#with tf.variable_scope(self.scope) as scope:
		#使用self.scope作为范围
		#均方根损失
		rms_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.gt_M, self.pre_M)))) #计算heatmap的损失函数（4）
		# tf.squared_difference(x, y, name=None) try this! 
		# (x-y)(x-y) 

		tf.add_to_collection('losses', rms_loss)


		lr = tf.train.exponential_decay(
			self.params['lr_initial'], 
			global_step, 
			self.params['lr_decay_steps'], 
			self.params['lr_decay_rate'] , 
			name='lr')

		# Vanilia SGD with dexp decay learning rate
		optimizer = tf.train.GradientDescentOptimizer(lr)

		if add_regulizer:
			#添加L2正则项
			total_losses = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), 'total_losses')
		else:
			total_losses = rms_loss
		train_op = optimizer.minimize(total_losses, var_list = self.variables, global_step=global_step)
		self.loss = total_losses
		return train_op, total_losses, lr, optimizer

	def sel_feature_maps(self, sess, vgg_conv, feed_dict, num_sel):#选择合适的feature map  4.1
		""" 
		Selects saliency feature maps. 
		The change of the Loss function by the permutation
		of the feature maps dF, can be computed by a 
		two-order Taylor expansions.

		Further simplication can be done by only compute
		the diagonol part of the Hessian matrix.

		S = - partial(L)/patial(F) * F 
			+ 0.5 * sencondOrderPartial(L)/F

		Args:
			gt_M: tensor, ground truth heat map.
			sel_maps: tensor, conv layer of vgg.
			num_sel: int, number of selected maps.

		Returns:
			sel_maps: np.ndarray, conv layer of vgg.
			idx: list, indexes of selected maps
		"""
		# 计算一阶导数（Lsel对feature——map里的偏导）
		grads = tf.gradients(self.loss, self.feature_maps)  #（5） gi  Lsel  fi

		# 使用黑森矩阵的对角矩阵进行二阶偏导  hij  Lsel fi fj
		# of Loss_x w.r.t x
		H_diag = [tf.gradients(grads[i], self.feature_maps[i])[0] for i in range(512)]


		#计算每个feature map里的各个元素的confidence  (6)si
		S = [tf.reduce_sum(-tf.mul(grads[i], self.feature_maps[i])) \
			+ 0.5 * tf.reduce_sum(tf.mul(H_diag[i], self.feature_maps[i]**2)) for i in range(512)]
		S_tensor = tf.pack(S, axis=0) # shape (512,)   Sk

		vgg_maps, signif_v = sess.run([vgg_conv, S_tensor], feed_dict=feed_dict)

		
		# 检索得分最高的feature maps和相应的id
		idxs = sorted(range(len(signif_v)), key=lambda i: signif_v[i])[-num_sel:]
		best_maps = vgg_maps[...,idxs]
		print('Selected maps shape:'%best_maps.shape)
		return best_maps, idxs
		
