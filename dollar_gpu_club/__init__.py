name = "dollar_gpu_club"

import os
import requests
import sys

import boto3
import botocore
import numpy as np
from keras.callbacks import Callback

DEV_MODE = os.environ.get('DEV_MODE', True)

s3 = None if DEV_MODE else boto3.resource('s3')
JOB_ID = 'dev_job_id' if DEV_MODE else os.environ['JOB_ID']
BUCKET_NAME = 'dev_bucket' if DEV_MODE else os.environ['BUCKET_NAME']
APP_DOMAIN = 'http://localhost:8080' if DEV_MODE else os.environ['APP_DOMAIN']
CHECKPOINT_FILE = '~/dev.h5' if DEV_MODE else '{}.h5'.format(JOB_ID)
ERROR_MSG = 'ERROR. JOB_ID: {}. BUCKET_NAME: {}. CHECKPOINT_FILE: {}.'.format(JOB_ID, BUCKET_NAME, CHECKPOINT_FILE)

def compile(model,
			optimizer,
			loss=None,
			metrics=None,
			loss_weights=None,
			sample_weight_mode=None,
			weighted_metrics=None,
			target_tensors=None):
	_load_checkpoint(model)
	return model.compile(
		optimizer,
		loss=loss,
		metrics=metrics,
		loss_weights=loss_weights,
		sample_weight_mode=sample_weight_mode,
		weighted_metrics=weighted_metrics,
		target_tensors=target_tensors,
	)

def fit(model,
		x=None,
		y=None,
		batch_size=None,
		epochs=1,
		verbose=1,
		callbacks=None,
		validation_split=0.0,
		validation_data=None,
		shuffle=True,
		class_weight=None,
		sample_weight=None,
		initial_epoch=0,
		steps_per_epoch=None,
		validation_steps=None):
	if DEV_MODE:
		print('skipping making a POST request to {}/{}/start'.format(APP_DOMAIN, JOB_ID))
	else:
		requests.post('{}/{}/start'.format(APP_DOMAIN, JOB_ID), data={})

	cbs = [MetricsCallback(), FinalCheckpointCallback()]
	cbs = cbs + callbacks if callbacks else cbs

	return model.fit(
		x=x,
		y=y,
		batch_size=batch_size,
		epochs=epochs,
		verbose=verbose,
		callbacks=cbs,
		validation_split=validation_split,
		validation_data=validation_data,
		shuffle=shuffle,
		class_weight=class_weight,
		sample_weight=sample_weight,
		initial_epoch=initial_epoch,
		steps_per_epoch=steps_per_epoch,
		validation_steps=validation_steps,
	)

def _load_checkpoint(model):
	if _checkpoint_exists():
		print('existing checkpoint found!')
		try:
			if DEV_MODE:
				print('skipping loading checkpoint: {}'.format(CHECKPOINT_FILE))
			else:
				s3.Bucket(BUCKET_NAME).download_file(JOB_ID, CHECKPOINT_FILE)
				model.load_weights(CHECKPOINT_FILE)
		except botocore.exceptions.ClientError as e:
			raise

def _checkpoint_exists():
	try:
		if DEV_MODE:
			print('skipping checking if there are existing checkpoints')
		else:
			s3.Object(BUCKET_NAME, CHECKPOINT_FILE).load()
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == '404':
			return False
		else:
			raise
	else:
		return True

class MetricsCallback(Callback):
	def __init__(self):
		super(MetricsCallback, self).__init__()

	def on_epoch_end(self, epoch, logs=None):
		loss, acc = logs.get('loss'), logs.get('acc')
		val_loss, val_acc = logs.get('val_loss'), logs.get('val_acc')
		if loss is None or acc is None:
			return
		data = {
			'epoch': epoch,
			'training': {
				'accuracy': acc,
				'loss': loss,
			},
		}
		validation = {}
		if val_loss:
			validation['loss'] = val_loss
		if val_acc:
			validation['accuracy'] = val_acc
		if len(validation) > 0:
			data['validation'] = validation
		if DEV_MODE:
			print('skipping sending data: {}'.format(data))
		else:
			requests.post('{}/{}/metrics'.format(APP_DOMAIN, JOB_ID), data=data)

# stolen from keras implementation
class FinalCheckpointCallback(Callback):
	def __init__(self):
		super(FinalCheckpointCallback, self).__init__()

	def _is_dying(self):
		if DEV_MODE:
			return False
		return requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code != 200

	def on_epoch_end(self, epoch, logs=None):
		if not self._is_dying():
			return
		self.model.save_weights(CHECKPOINT_FILE, overwrite=True)
		if DEV_MODE:
			print('skipping saving checkpoint file {} to s3'.format(CHECKPOINT_FILE))
		else:
			print('uploading checkpoint file {} to s3'.format(CHECKPOINT_FILE))
			s3.meta.client.upload_file(CHECKPOINT_FILE, BUCKET_NAME, CHECKPOINT_FILE)
			requests.post('{}/{}/halt'.format(APP_DOMAIN, JOB_ID), data={})
		sys.exit()
