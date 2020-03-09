#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import cPickle as pickle
import threading
import Queue
import sys
from skimage import io,transform
import pandas as pd
import os

class DataFetcher(threading.Thread):
	def __init__(self, data_dir, class_file):
		super(DataFetcher, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)
		self.data_dir = data_dir

		meta_path = os.path.join(data_dir, class_file)
		classes_dt = pd.read_csv(meta_path, delimiter=',')
		self.labels = {k: i for i, k in enumerate(np.squeeze(classes_dt[['classname']].values, axis=1).tolist())}
		self.class_files = np.squeeze(classes_dt[['filename']].values, axis=1).tolist()

		self.file_names = []
		for file in self.class_files:
			file_path = os.path.join(data_dir, file)
			files = pd.read_csv(file_path, delimiter=',')
			self.file_names.extend(np.squeeze(files.values, axis=1).tolist())

		self.index = 0
		self.number = len(self.file_names)
		np.random.shuffle(self.file_names)

	def work(self, idx):
		filename = self.file_names[idx]
		image_path = os.path.join(self.data_dir, filename)
		pts = pd.read_csv(image_path.replace('.png', '.xyz')).to_numpy().astype(np.float32)
		pts[:, 3] -= np.array([0.0, 0.0, 0.8])
		img = io.imread(image_path)
		img[np.where(img[:, :, 3] == 0)] = 255
		img = transform.resize(img, (224, 224))
		img = img[:, :, :3].astype('float32')

		return img, pts, filename
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.file_names)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

if __name__ == '__main__':
	file_list = sys.argv[1]
	data = DataFetcher(file_list)
	data.start()

	image,point,normal,_,_ = data.fetch()
	print image.shape
	print point.shape
	print normal.shape
	data.stopped = True
