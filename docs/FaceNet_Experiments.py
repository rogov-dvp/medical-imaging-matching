import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from triplet_loss_function import triplet_loss

IMAGE_DIM = 160

def preprocess(path):
	img = Image.open(path)
	img = img.resize((IMAGE_DIM, IMAGE_DIM))
	return img

# load images 
def load_mammorgams(directory):
	mams = []
	# enumerate files
	for filename in os.listdir(directory):
		# path
		path = directory + filename
		# get face
		mam = preprocess(path)
		# store
		mams.append(mam)
	return mams


# create pos. by augmenting the anchor
def augment(img):
	img = img.rotate(1)
	return img


# load the model
model = load_model('facenet_keras.h5')
model.load_weights('facenet_keras_weights.h5')

# compile the model
model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
model.summary()


# train the model train and validate need to be defined
# history = model.fit(
#     train,
#     validation_data = validate,
#     steps_per_epoch = 100,
#     epochs = 1,
#     validation_steps = 50,
#     verbose = 2)

anc = np.array(preprocess('Bond2.jpg'))
pos = np.array(augment(preprocess('Bond3.jpg')))
neg = np.array(preprocess('Bond1.jpeg'))

anc = np.expand_dims(anc, 0)
pos = np.expand_dims(pos, 0)
neg = np.expand_dims(neg, 0)

anc_pred = model.predict(anc)
pos_pred = model.predict(pos)
neg_pred = model.predict(neg)

dist_pos = np.linalg.norm(np.subtract(anc_pred, pos_pred))
dist_neg = np.linalg.norm(np.subtract(anc_pred, neg_pred))

print(f'dist Anchor and Positive Bond: {dist_pos}')
print(f'dist Anchor and Negative Bond: {dist_neg}')
