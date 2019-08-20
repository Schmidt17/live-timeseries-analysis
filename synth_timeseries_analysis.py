import numpy as np
import multiprocessing
from multiprocessing import Process, Pipe
import pyqtgraph as pg

class KeyPressWindow(pg.GraphicsWindow):
	"""Subclass pyqtgraph's GraphicsWindow to add a signal for a key being pressed."""
    sigKeyPressed = pg.QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPressed.emit(ev)

def keyPressHandler(ev):
	if ev.key() == 68: # key "d"
		# toggle detecting
		global detecting
		detecting = not detecting
		if detecting:
			print("Started detecting ...")
		else:
			print("Stopped detecting.")

def start_event():
	""" Start the timer of a spike event """
	global event_t
	event_t = 0

def print_prediction(pipe):
	"""
	Performs the classification. 
	This is run in a separate process and encapsulates all tensorflow operations.
	Data is fed from the main process over a multiprocessing Pipe and the classification
	results are returned via the same Pipe.
	"""
	import tensorflow as tf
	tf.TF_CPP_MIN_LOG_LEVEL=2
	tf.enable_eager_execution()
	model = tf.keras.models.load_model("spike_detector_units16_windowsize17.h5")
	print("")
	print("Ready to detect!")

	while True:
		samples, start_times = pipe.recv()
		prediction = np.round(model.predict(samples))

		if np.any(prediction):
			prediction = prediction.flatten()
			first_detection_index = np.where(prediction == 1.)[0][0]
			pipe.send(start_times[first_detection_index])
			print("Spike detected!")

if __name__ == "__main__":
	window_size = 160

	data = np.zeros(window_size, dtype=np.float32)
	spike = np.array([0., 20., 40., 60., 40., 20., 0., -20., -40., -60., -40., -20., 0.], dtype=np.float32)

	multiprocessing.set_start_method('spawn')
	parent_conn, child_conn = Pipe()
	pred_process = Process(target=print_prediction, args=(child_conn,))
	pred_process.start()

	event_t = spike.size
	t = 0
	training = False
	detecting = False
	sample_buffer = []
	start_times_buffer = []
	buffer_size = 10
	sample_window = 17
	last_detected_time = 0
	refractory_period = 24

	win = KeyPressWindow()
	plot_obj = win.addPlot()
	main_curve = plot_obj.plot(data)
	plot_obj.setYRange(-100, 100)
	win.scene().sigMouseClicked.connect(start_event)
	win.sigKeyPressed.connect(keyPressHandler)

	spike_onsets = []  # store some ROI rectangles for spike marking
	spike_pen = pg.mkPen('r', width=2)
	spike_curves = []

	while True:
		sample = 20 * (np.random.random() - 0.5)  # create a new random data point between -10 and 10

		if event_t < spike.size:  # if a spike event has been triggered by clicking the mouse
			sample += spike[event_t]
			event_t += 1

		data[:-1] = np.copy(data[1:])
		data[-1] = sample

		if detecting:
			first_detected_time = None
			if parent_conn.poll():
				first_detected_time = parent_conn.recv()
				if not first_detected_time - last_detected_time < refractory_period:
					spike_onsets.append(data.size - (t - first_detected_time))
					spike_curves.append(plot_obj.plot(np.arange(spike_onsets[-1], spike_onsets[-1]+sample_window), data[spike_onsets[-1]: spike_onsets[-1] + sample_window], pen=spike_pen))
					last_detected_time = first_detected_time
				else:
					print("\tSpike rejected!")

			if len(sample_buffer) < buffer_size:
				sample_buffer.append(np.copy(data[-sample_window:]).reshape((-1, 1)))
				start_times_buffer.append(t - sample_window)
			else:
				parent_conn.send((np.array(sample_buffer), np.array(start_times_buffer)))
				sample_buffer = []
				start_times_buffer = []

		main_curve.setData(data)
		for i, roi in enumerate(spike_onsets):
			spike_curves[i].setData(np.arange(roi, roi+sample_window), data[roi: roi + sample_window])

		pg.QtGui.QApplication.processEvents()

		t += 1
		for i in reversed(range(len(spike_onsets))):
			if spike_onsets[i] == 0:
				spike_onsets.pop(i)
				plot_obj.removeItem(spike_curves[i])
				spike_curves.pop(i)
			else:
				spike_onsets[i] = spike_onsets[i] - 1
