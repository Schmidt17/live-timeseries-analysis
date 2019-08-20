import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# spike of size 13
spike = np.array([0., 20., 40., 60., 40., 20., 0., -20., -40., -60., -40., -20., 0.], dtype=np.float32)

def generate_sample(class_label, length):
	sample = 20 * (np.random.random(length) - 0.5)

	if class_label == 1:
		start_t = length//2-spike.size//2 + (np.random.randint(4) - 2)
		sample[start_t:start_t + spike.size] += spike

	return sample

N_train = 100
N_test = 50
sample_length = 17

trainY = np.random.randint(2, size=N_train)
trainX = np.array([generate_sample(c, sample_length).reshape((-1, 1)) for c in trainY])

testY = np.random.randint(2, size=N_test)
testX = np.array([generate_sample(c, sample_length).reshape((-1, 1)) for c in testY])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=2, batch_size=1, validation_data=(testX, testY))

# Save the model to a file for later usage
model.save("spike_detector_units16_windowsize17.h5")

pred = model.predict(testX[0].reshape((1, -1, 1)))
print(pred.flatten())
print("Actual:")
print(testY[0])

print("Done.")