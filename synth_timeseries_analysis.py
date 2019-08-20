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
    global spike_type
    spike_type = np.random.randint(1, 3)

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
    model = tf.keras.models.load_model("spike_detector_classes3_units16_windowsize17.h5")
    print("")
    print("Ready to detect!")

    while True:
        samples, start_times = pipe.recv()  # wait for new batch of samples to arrive

        estimates = model.predict(samples)  # get the model class probabilities (output shape (batch_size, num_classes))
        prediction = np.argmax(estimates, axis=-1)  # get the class predictions (1D, shape (batch_size,))

        spike_candidates = np.where(prediction != 0)[0]  # now look at all samples which get classified as not class 0 (background noise)
        if spike_candidates.size > 0:  # if there are any:
            max_prediction = np.max(estimates[spike_candidates], axis=-1)  # get the highest class probability for each non-noise sample (shape (spike_candidates.size,)) ...
            strongest_pred_pos = np.argmax(max_prediction)  # ... and check which one has the highest probability for whatever class it happens to be
            if max_prediction[strongest_pred_pos] > 0.9:  # if the confidence is above 90%, we accept it as a detection!
                accepted_sample_index = spike_candidates[strongest_pred_pos]  # make sure to convert the only-non-noise-index "strongest_pred_pos" back to the overall-sample-index by feeding it back into "spike_candidates"
                pipe.send((start_times[accepted_sample_index], prediction[accepted_sample_index]))
                print("Spike of type {0} detected, confidence {1:.3f}!".format(prediction[accepted_sample_index], max_prediction[strongest_pred_pos]))

if __name__ == "__main__":
    window_size = 160

    data = np.zeros(window_size, dtype=np.float32)
    # spike of size 13
    spike = np.array([0., 20., 40., 60., 40., 20., 0., -20., -40., -60., -40., -20., 0.], dtype=np.float32)
    # reverse spike of size 13
    reverse_spike = np.array(list(reversed([0., 20., 40., 60., 40., 20., 0., -20., -40., -60., -40., -20., 0.])), dtype=np.float32)

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

    spike_onsets = []  # store the detected spike onsets for spike marking in the plot
    spike_pen = pg.mkPen('r', width=2)
    reverse_spike_pen = pg.mkPen('b', width=2)
    spike_curves = []
    spike_type = 1  # 1: first up, then down, 2: first down, then up

    while True:
        sample = 20 * (np.random.random() - 0.5)  # create a new random data point between -10 and 10

        if event_t < spike.size:  # if a spike event has been triggered by clicking the mouse
            if spike_type == 1:
                sample += spike[event_t]
            else:
                sample += reverse_spike[event_t]
            event_t += 1

        data[:-1] = np.copy(data[1:])
        data[-1] = sample

        if detecting:
            first_detected_time = None
            if parent_conn.poll():
                first_detected_time, detected_spike_type = parent_conn.recv()
                if not first_detected_time - last_detected_time < refractory_period:
                    spike_onsets.append(data.size - (t - first_detected_time))
                    if detected_spike_type == 1:
                        spike_curves.append(plot_obj.plot(np.arange(spike_onsets[-1], spike_onsets[-1]+sample_window), data[spike_onsets[-1]: spike_onsets[-1] + sample_window], pen=spike_pen))
                    else:
                        spike_curves.append(plot_obj.plot(np.arange(spike_onsets[-1], spike_onsets[-1]+sample_window), data[spike_onsets[-1]: spike_onsets[-1] + sample_window], pen=reverse_spike_pen))
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

        t += 1  # advance the global time

        # move the spike onset markers to the left to follow the moving plot
        # remove them if they touch the left boundary (at index 0)
        for i in reversed(range(len(spike_onsets))):
            if spike_onsets[i] == 0:
                spike_onsets.pop(i)
                plot_obj.removeItem(spike_curves[i])
                spike_curves.pop(i)
            else:
                spike_onsets[i] = spike_onsets[i] - 1
