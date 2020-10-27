import numpy as np
import tensorflow as tf
import cv2


class ImageGenerator:
	def __init__(self, min_digs_count:int=3, max_digs_count:int=8,
		generation_type:str='mnist', shape:tuple=(500, 500, 3), background_type:str='constant',
		background_fill:tuple=(0,0,0), background_image:np.array=None, random_size:bool=True,
		min_resize_factor:int=1, max_resize_factor:int=4, augs=None, batch_size:int=5):


		assert background_type in ['constant', 'image'], 'background_type can be \'constant\' or \'image\''
		assert generation_type in ['mnist'], 'generation_type can be mnist'

		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		self.digs = np.concatenate([x_train, x_test])
		self.labels = np.concatenate([y_train, y_test])

		self.min_digs_count = min_digs_count
		self.max_digs_count = max_digs_count
		self.generation_type = generation_type
		self.shape = shape
		self.background_type = background_type
		self.background_fill = background_fill
		self.background_image = background_image
		self.random_size = random_size
		self.min_resize_factor = min_resize_factor
		self.max_resize_factor = max_resize_factor
		self.augs = augs

		self.blocks = []
		for y in range(0,shape[0] - self.max_resize_factor*digs.shape[1], self.max_resize_factor*digs.shape[1]):
			for x in range(0,shape[1] - self.max_resize_factor*digs.shape[2], self.max_resize_factor*digs.shape[2]):
				self.blocks.append([y,y+digs.shape[1], x, x+digs.shape[2]])




	def generate_single(self, background_image=None):
		if self.background_type == 'constant':
			bg = np.full(shape=self.shape, fill_value=self.background_fill)
		elif self.background_type == 'image':
			if background_image is None:
		  		raise Exception('Background image is None!')
			bg = background_image.copy()

		mask = np.zeros(shape=(self.shape[0], self.shape[1], 1))

		local_blocks = self.blocks.copy()
		digs_count = min(np.random.randint(self.min_digs_count, self.max_digs_count), len(self.blocks))

		for _ in range(digs_count):
		rnd_img_idx = np.random.randint(0, len(digs))
		img = self.digs[rnd_img_idx]

		f = np.random.uniform(self.min_resize_factor, self.max_resize_factor)
		img = cv2.resize(img,None,fx=f,fy=f)
		img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

		label = labels[rnd_img_idx]

		rnd_block_idx = np.random.randint(0, len(local_blocks))
		block = local_blocks[rnd_block_idx]
		del local_blocks[rnd_block_idx]

		transformed = self.augs(image=img_colored)
		img_colored = transformed['image']


		bg[block[0]:block[0] + img_colored.shape[0], block[2]:block[2] + img_colored.shape[1]] = img_colored


		img_mask = np.expand_dims((img > 127).astype(int)*(label + 1), axis=-1)
		mask[block[0]:block[0] + img_colored.shape[0], block[2]:block[2] + img_colored.shape[1]] = img_mask

		return bg, mask

