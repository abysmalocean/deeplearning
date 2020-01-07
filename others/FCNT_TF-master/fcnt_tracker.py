"""
Main script for FCNT tracker. 
"""
#%%
# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from selcnn import SelCNN
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval

import numpy as np 
import tensorflow as tf

import os

#%%
tf.app.flags.DEFINE_integer('iter_step_sel', 200,
                          """Number of steps for trainning"""
                          """selCNN networks.""")
tf.app.flags.DEFINE_integer('iter_step_sg', 50,
                          """Number of steps for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('num_sel', 384,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_integer('iter_max', 200,
							"""Max iter times through imgs""")

FLAGS = tf.app.flags.FLAGS

## Define varies path
DATA_ROOT = 'data/Dog1'
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'


def train_selCNN(sess, selCNN, gt_M_sz, feed_dict):
	#初始化参数变量
	global_step = tf.Variable(0, trainable=False)
	selCNN_vars = selCNN.variables 
	init_vars_op = tf.initialize_variables(selCNN_vars + [global_step], name='init_selCNN')
	sess.run(init_vars_op)


	#检索培训操作
	train_op, losses, lr, optimizer = selCNN.train_op(gt_M_sz, global_step)
	print(sess.run(tf.report_uninitialized_variables()))

	#训练iter_step_sel次
	#检查损失曲线和pre——M
	for step in range(FLAGS.iter_step_sel):
		_, total_loss, lr_ = sess.run([train_op, losses, lr], feed_dict=feed_dict)
		print(total_loss)


def train_sgNet(sess, gnet, snet, sgt_M, ggt_M, feed_dict):
	"""
	Train sgnet by minimize the loss
	Loss = Lg + Ls
	where Li = |pre_Mi - gt_M|**2 + Weights_decay_term_i

	"""

	#初始化SNET,GNET的变量
	sgNet_vars = gnet.variables + snet.variables #变量是Snet+Gnet的变量总和
	init_SGNet_vars_op = tf.initialize_variables(sgNet_vars, name='init_sgNet')
	sess.run(init_SGNet_vars_op)

	# Define composite loss  定义损失函数为SNET+GNET损失函数之和
	total_losses = snet.loss(sgt_M) + gnet.loss(ggt_M)  #(7)

	# Define trainning op  定义学习率为-6  文章里的学习率时-7   P7
	optimizer = tf.train.GradientDescentOptimizer(1e-6)
	train_op = optimizer.minimize(total_losses, var_list= sgNet_vars)  #定义训练目标，使total_losses最小化

	for step in range(FLAGS.iter_step_sg):
		loss, _ = sess.run([total_losses, train_op], feed_dict = feed_dict)
		print(loss)



def gen_mask_phi(img_sz, loc):
	x,y,w,h = loc
	phi = np.zeros(img_sz)
	phi[[y-int(0.5*h): y+int(0.5*h), x-int(0.5*w):x+int(0.5*w)]] = 1
	return phi


def main(args):

	#实例化imputProducer并且检索具有ground Truth的第一帧图片
	inputProducer = InputProducer(IMG_PATH, GT_PATH)
	img, gt, t  = next(inputProducer.gen_img)
	roi_t0, _, _ = inputProducer.extract_roi(img, gt)


	#预测第一幅图片
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
	vgg.print_prob(roi_t0, sess)


	#t=0时，按以下进行训练
	#1.用局部feature map 和全局feature map 训练selCNN
	#2.训练SNET,GNET
	assert t == 0:

	lselCNN = SelCNN('sel_local', vgg.conv4_3)
	gselCNN = SelCNN('sel_global', vgg.con5_3)


	# 对目标区域生成标注的mask
	sgt_M = inputProducer.gen_mask(lselCNN.pre_M_size)
	ggt_M = inputProducer.gen_mask(gselCNN.pre_M_size)


	#用第一帧ROI训练selCNN 4/7
	#重新调整大小
	sgt_M = sgt_M[np.newaxis,:,:,np.newaxis]
	ggt_M = ggt_M[np.newaxis,:,:,np.newaxis]

	feed_dict = {vgg.imgs: [roi_t0], 
				lselCNN.gt_M: sgt_M,
				gselCNN.gt_M: ggt_M} # corrpus the other nets?

	train_selCNN(sess, lselCNN, sgt_M.shape, feed_dict)
	train_selCNN(sess, gselCNN, ggt_M.shape, feed_dict)

	# 执行显著性选择
	s_sel_maps, s_idx = lselCNN.sel_feature_maps(sess, vgg.conv4_3, feed_dict,FLAGS.num_sel)
	g_sel_maps, g_idx = gselCNN.sel_feature_maps(sess, vgg.conv5_3, feed_dict,FLAGS.num_sel)

	assert isinstance(s_sel_maps, np.ndarray)
	assert isinstance(g_sel_maps, np.ndarray)
	assert len(s_sel_maps.shape) == 4


	# 实例化SNET,GNET
	gnet = GNet('GNet', s_sel_maps.shape)
	snet = SNet('SNet', s_sel_maps.shape)


	#为每个图像提供选定的显著性图来最小化复合损失来训练SN T,GNET
	feed_dict = {gnet.input_maps: g_sel_maps, snet.input_maps: s_sel_maps}
	train_sgNet(sess, gnet, snet, sgt_M, ggt_M, feed_dict)
	s_sel_maps_t0 = s_sel_maps


	
	
	#当t>0的时候，每帧执行目标定位和干扰检测，每20帧执行SNET的自适应更新，如果干扰检测返回true，则执行SNET判断更新
	#实例化Tracker对象并使用sgt_M对其进行初始化。
	tracker = TrackerVanilla(sgt_M, gt)

	# Iter imgs  6/7
	gt_last = gt 
	for i in range(FLAGS.iter_max):
		# 对下一帧图片进行信息产生 
		img, gt_cur, t  = next(inputProducer.gen_img)

		## 从上一帧图像中对感兴趣区域进行裁剪
		roi, _, resize_factor = inputProducer.extract_roi(img, gt_last)
		
		
		#将GNET预测得到的目标区域进行展示，并且产生GNet的热力图
		feed_dict_vgg = {vgg.imgs : [roi]}
		s_maps, g_maps = sess.run([vgg.conv4_3, vgg.conv5_3], feed_dict=feed_dict_vgg)
		s_sel_maps = s_maps[s_idx] # np.ndarray, shape = [1,28,28,num_sel]?
		g_sel_maps = g_maps[g_idx]

		feed_dict_g = { gnet.input_maps: g_sel_maps}
		pre_M = sess.run(gnet.pre_M, feed_dict=feed_dict_g)
		tracker.pre_M_q.push(pre_M)

		if i % 20 == 0:
			
			#在中间帧内检索confidence最高的结果
			best_M = tracker.gen_best_M()

			
			#用预测效果最好的热力图微调SNET
			snet.adaptive_finetune(sess, best_M)


		#使用蒙特卡洛采样来定位目标
		tracker.draw_particles()
		pre_loc = tracker.predict_location(pre_M, gt_last, resize_factor, t)

		
		# 进行目标检测效果的检测
		if tracker.distracted():
			# 如果监测到干扰因素，则更新
			# SNet 使用descrimtive loss，第二种SNET更新方式
			# 产生mask
			phi = gen_mask_phi(roi.shape, pre_loc)
			snet.descrimtive_finetune(sess, s_sel_maps_t0, sgt_M, roi, s_sel_maps, phi)
			pre_M = sess.run(snet.pre_M, feed_dict=feed_dict)

			
			#若需要使用SNET,则用SNET来进行位置预测
			pre_loc = tracker.predict_location(pre_M)
		
		
		#使用预测的位置窗口作为下一帧的ground truth
		gt_last = pre_loc

		
		#将bounding box 显示在图片上，并且加上IOU分数
		img_with_bbox(img, pre_loc, gt_cur)
		IOU_eval()

if __name__=='__main__':
	tf.app.run()

