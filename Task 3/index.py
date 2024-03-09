import wave
import scipy
import sounddevice as sd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pyqtgraph import PlotWidget
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import numpy as np
import pyqtgraph as pg
import pandas as pd
import os
import random
import sys
import  math
from os import path
from reportlab.lib import colors
from reportlab.platypus import *
from PyQt5.QtWidgets import QMessageBox
import scipy.fftpack as fft
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QObject, pyqtSignal
import scipy.interpolate as interp
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PyQt5.QtCore import QTimer
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import tempfile

import soundfile as sf
# import librosa
from PyQt5.QtWidgets import QShortcut



CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ui,_=loadUiType('untitled.ui')
ui_popup, _ = loadUiType('popup.ui')  # Load the popup dialog UI

# Define a global variable
global_variable = 1
class MainApp(QMainWindow, ui):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)
        # 7agat el signal
        self.figures = {}
        self.signal_data = []
        self.x_values = []
        self.original_signal_data = []
        self.audio_data = None  # Initialize audio data variable
        self.mode_index = 0
        self.currentIndex=0
        self.inv_transform=[]
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_audio_files')
        os.makedirs(self.temp_dir, exist_ok=True)  # Create the directory if it doesn't exist

        #play button
        self.play_pause_btn = self.findChild(QPushButton, 'play_pause_btn')
        self.play_pause_btn.clicked.connect(self.play_pause_btn_clicked)
        self.before_first_time = True
        self.before_playing = False
        self.after_first_time = True
        self.after_playing = False


        # Create a QMediaPlayer instance
        self.media_player_before = QMediaPlayer()
        self.media_player_after = QMediaPlayer()
        self.before_or_after = self.findChild(QComboBox, 'before_or_after_comboBox')
        self.before_or_after.currentIndexChanged.connect(self.on_combobox_changed)

        #Arrays for band frequencies
        self.list1=[]
        self.list2=[]
        self.list3=[[600,800],[3800,7000],[1000,6000],[0,3000]]

        #ezay teshow we tehide
        self.sliders_arr = []
        self.labels_arr = []
        self.mode_combo_box = self.findChild(QComboBox,'select_mode')
        self.mode_combo_box.currentIndexChanged.connect(self.handle_combobox)
        self.import_btn = self.findChild(QPushButton, 'import_btn')
        self.import_btn.clicked.connect(lambda :self.import_btn_clicked())
        for i in range(1, 11):
            slider = self.findChild(QSlider, f'verticalSlider_{i}')
            label = self.findChild(QLabel, f'label_{i}')
            # Set the range of the slider to be from 0 to 5
            slider.setRange(0, 5)
            # Set the initial value of the slider to be 1
            slider.setValue(1)
            self.sliders_arr.append(slider)
            self.labels_arr.append(label)
        for index, slider in enumerate(self.sliders_arr):
            slider.valueChanged.connect(lambda value, idx=index: self.slider_value_changed(value, idx))
        # ezay teswitch tabs
        self.go_to_smoothing = self.findChild(QPushButton, 'go_to_smoothing')
        self.go_to_smoothing.clicked.connect(self.show_popup)

        before_spectro_widget = self.findChild(QWidget, 'before_spectro_widget')
        widget_layout = QVBoxLayout(before_spectro_widget)
        widget_layout.addWidget(self.before_spectro_widget)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        self.handle_combobox(0)
        self.before_or_after.setCurrentIndex(1)

        # after spectro widget
        after_spectro_widget = self.findChild(QWidget, 'after_spectro_widget')
        widget_layout = QVBoxLayout(after_spectro_widget)
        widget_layout.addWidget(self.after_spectro_widget)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        # Connecting all main window widgets
        self.before_time_widget = pg.PlotWidget()
        before_time_widget = self.findChild(QWidget, 'before_time_widget')
        before_time_layout = QVBoxLayout(before_time_widget)
        before_time_layout.addWidget(self.before_time_widget)
        before_time_layout.setContentsMargins(0, 0, 0, 0)

        self.after_time_widget = pg.PlotWidget()
        after_time_widget = self.findChild(QWidget, 'after_time_widget')
        after_time_widget = QVBoxLayout(after_time_widget)
        after_time_widget.addWidget(self.after_time_widget)
        after_time_widget.setContentsMargins(0, 0, 0, 0)

        self.frequency_widget = pg.PlotWidget()
        frequency_widget = self.findChild(QWidget, 'frequency_widget')
        frequency_widget = QVBoxLayout(frequency_widget)
        frequency_widget.addWidget(self.frequency_widget)
        frequency_widget.setContentsMargins(0, 0, 0, 0)

        self.smoothing_window_index=1


        # Create an InfiniteLine (marker line) and add it to the PlotWidget
        self.marker_line = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen('r', width=2))
        self.before_time_widget.addItem(self.marker_line)  # Add the marker line to the plot

        # 7agat spectrogram
        self.show_spectro_checkBox = self.findChild(QCheckBox, 'show_spectro_checkBox')
        self.show_spectro_checkBox.stateChanged.connect(self.handle_spectro_check_box)
        self.fig, self.ax = plt.subplots()
        self.before_spectro_widget.hide()
        self.after_spectro_widget.hide()
        #speed slider
        self.speed_Slider= self.findChild(QSlider, 'speed_Slider')
        self.speed_Slider.setRange(25, 200)
        self.speed_Slider.setSingleStep(25)
        self.speed_Slider.setPageStep(25)
        self.speed_Slider.setValue(100)
        self.speed_Slider.valueChanged.connect(self.speed_slider_changed)
        #shortcuts for zoom and reset
        zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        reset_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        zoom_in_shortcut.activated.connect(lambda: self.zoom("in"))
        zoom_out_shortcut.activated.connect(lambda: self.zoom("out"))
        reset_shortcut.activated.connect(self.reset_view)
        #Replay button
        self.replay_btn = self.findChild(QPushButton, 'replay_btn')
        self.replay_btn.clicked.connect(self.replay)



    def show_popup(self):
        self.popup = PopupDialog()  # Create an instance of the popup dialog
        self.popup.show()  # Show the popup dialog as a modal dialog
        self.popup.updateSmoothedGraph(0)


    def hide_sliders(self):
        total_sliders = len(self.sliders_arr)
        for i in range(total_sliders - 6, total_sliders):
            self.sliders_arr[i].hide()
            self.labels_arr[i].hide()

    def show_sliders(self):
        total_sliders = len(self.sliders_arr)
        for i in range(total_sliders - 6, total_sliders):
            self.sliders_arr[i].show()
            self.labels_arr[i].show()

    def set_label_font(self, label, text, font_size):
        label.setText(text)
        label.setFont(QFont("Arial", font_size))

    #set sliders and labels according to selected mode
    def handle_combobox(self,index):
        self.mode_index=index
        for slider in self.sliders_arr:
            # Set the initial value of the slider to be 1
            slider.setValue(1)
        if index == 0:
            self.show_sliders()
            font_size = 10
            for i in range(10):
                self.set_label_font(self.labels_arr[i], str(i + 1), font_size)

        elif index == 1:
            self.hide_sliders()
            font_size = 18
            self.set_label_font(self.label_1, "🎻 ", font_size)
            self.set_label_font(self.label_2, "🎺", font_size)
            self.set_label_font(self.label_3, "🎹 ", font_size)
            self.set_label_font(self.label_4, "🔔", font_size)

        elif index == 2:
            self.hide_sliders()
            font_size = 10
            self.set_label_font(self.label_1, "Normal ", font_size)
            self.set_label_font(self.label_2, "Arrhythmia_1", font_size)
            self.set_label_font(self.label_3, "Arrhythmia_2", font_size)
            self.set_label_font(self.label_4, "Arrhythmia_3", font_size)


        elif index == 3:
            self.hide_sliders()
            font_size = 18
            self.set_label_font(self.label_1, "🐋 ", font_size)
            self.set_label_font(self.label_2, "🦗", font_size)
            self.set_label_font(self.label_3, "🐦", font_size)
            self.set_label_font(self.label_4, "🐶", font_size)

    def play_pause_btn_clicked(self):
        if self.currentIndex == 1:
            self.play_before()
        elif self.currentIndex == 2:
            self.play_after()
        else:
            self.play_before()
            self.play_after()

    def on_combobox_changed(self, index):
        self.currentIndex=index

    def play_before(self):
        try:
            if self.new_audio_data is not None:
                if self.before_first_time:
                    # Save audio data to a temporary file in the temporary directory
                    temp_file = os.path.join(self.temp_dir, 'temp_audio.wav')
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    scipy.io.wavfile.write(temp_file, self.sample_rate, self.new_audio_data)
                    # Create QMediaContent with the audio file path
                    del self.media_player_before
                    self.media_player_before = QMediaPlayer()
                    media_content = QMediaContent(QUrl.fromLocalFile(temp_file))
                    # Set the media content to QMediaPlayer and play
                    self.media_player_before.setMedia(media_content)
                    self.media_player_before.play()
                    self.play_pause_btn.setText('Pause')
                    self.before_first_time = False
                else:
                    if self.media_player_before.state() == QMediaPlayer.PlayingState:
                        print("sha8aaaaal")
                        self.play_pause_btn.setText('Play')
                        self.media_player_before.pause()
                    else:
                        self.media_player_before.play()
                        self.play_pause_btn.setText('Pause')
            else:
                print("No audio data available.")
        except Exception as e:
            print(f'Error in before_btn_clicked: {e}')

    def play_after(self):
        try:
            if self.after_first_time:
                if self.inv_transform is not None:
                    # Perform inverse Fourier transform
                    # Save transformed audio data to a temporary file in the temporary directory
                    temp_file = os.path.join(self.temp_dir, 'temp_transformed_audio.wav')
                    self.ifft(self.copy_freq_signal_data)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    scipy.io.wavfile.write(temp_file, self.sample_rate, self.inv_transform.real.astype(np.int16))
                    # Delete the existing media player
                    del self.media_player_after
                    # Create a new media player
                    self.media_player_after = QMediaPlayer()
                    # Create QMediaContent with the temporary transformed audio file path
                    media_content = QMediaContent(QUrl.fromLocalFile(temp_file))
                    # Set the media content to QMediaPlayer and play
                    self.media_player_after.setMedia(media_content)
                    self.media_player_after.play()
                    self.play_pause_btn.setText('Pause')
                    self.after_first_time = False
                else:
                    print("No transformed data available.")
            else:
                if self.media_player_after.state() == QMediaPlayer.PlayingState:
                    print("sha8aaaaal2")
                    self.play_pause_btn.setText('Play')
                    self.media_player_after.pause()
                else:
                    self.media_player_after.play()
                    self.play_pause_btn.setText('Pause')


        except Exception as e:
            print(f'Error in after_btn_clicked: {e}')


    def replay(self):
        self.media_player_before.setPosition(0)  # Set position to the beginning
        self.media_player_before.play()
        self.play_pause_btn.setText('Pause')
        self.media_player_before.positionChanged.connect(self.update_marker_position)
        self.before_first_time = True
        self.after_first_time = True

    def speed_slider_changed(self):
        index = self.before_or_after.currentIndex()
        if index == 1:
            if self.media_player_before.state() == QMediaPlayer.PlayingState:
                speed_factor = self.speed_Slider.value() / 100
                self.media_player_before.setPlaybackRate(speed_factor)
        elif index == 2:
            if self.media_player_after and self.media_player_after.state() == QMediaPlayer.PlayingState:
                speed_factor = self.speed_Slider.value() / 100
                self.media_player_after.setPlaybackRate(speed_factor)


    def zoom(self, direction):
        index = self.before_or_after.currentIndex()
        if direction == "in":
            factor = 0.8
        elif direction == "out":
            factor = 1.25
        else:
            return
        if index == 1:
            widget = self.before_time_widget
        elif index == 2:
            widget = self.after_time_widget
        elif index == 0:
            current_x_min, current_x_max = self.before_time_widget.viewRange()[0]
            new_x_min = current_x_min + (current_x_max - current_x_min) * factor
            new_x_max = current_x_max - (current_x_max - current_x_min) * factor
            self.before_time_widget.setXRange(new_x_min, new_x_max)
            current_x_min, current_x_max = self.after_time_widget.viewRange()[0]
            new_x_min = current_x_min + (current_x_max - current_x_min) * factor
            new_x_max = current_x_max - (current_x_max - current_x_min) * factor
            self.after_time_widget.setXRange(new_x_min, new_x_max)
            return
        current_x_min, current_x_max = widget.viewRange()[0]
        new_x_min = current_x_min + (current_x_max - current_x_min) * factor
        new_x_max = current_x_max - (current_x_max - current_x_min) * factor
        widget.setXRange(new_x_min, new_x_max)

    def reset_view(self):
        index = self.before_or_after.currentIndex()
        original_x_min = 0
        original_x_max = self.x_max_range
        if index == 1:
            self.before_time_widget.setXRange(original_x_min, original_x_max)
        elif index == 2:
            self.after_time_widget.setXRange(original_x_min, original_x_max)
        elif index == 0:
            self.before_time_widget.setXRange(original_x_min, original_x_max)
            self.after_time_widget.setXRange(original_x_min, original_x_max)



    def update_marker_position(self, position):
        if self.audio_data is not None:
            duration = self.media_player_before.duration()
            ratio = position / duration if duration > 0 else 0
            num_frames = len(self.audio_data)
            current_frame = int(ratio * num_frames)
            time_values = np.linspace(0, duration / 1000, num_frames)  # Assuming time values
            current_time = time_values[current_frame]
            self.marker_line.setPos(current_time)

    def import_btn_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog_filters = "CSV Files (*.csv);;WAV Files (*.wav);;All Files (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", file_dialog_filters, options=options)

        if files:
            for file_path in files:
                try:
                    if file_path.endswith('.csv'):
                        # Read CSV file
                        df = pd.read_csv(file_path)
                        # Get the time and amplitude values from the file
                        y_values = df.iloc[:, -1].values
                        x_values = df.iloc[:, 0]
                        self.signal_data = y_values
                        self.x_values = x_values
                        self.plot_from_csv(x_values, y_values, self.before_time_widget)
                        Fs = 1 / (self.x_values[1] - self.x_values[0])
                        self.Fs = np.ceil(Fs)
                        self.create_spectrogram(self.signal_data, self.before_spectro_widget)
                        self.show_spectro_checkBox.setChecked(True)
                        self.time_step = 1 / self.Fs
                        self.N = len(self.signal_data)
                        # Compute the FFT of the signal
                        self.fft()
                        # Select the frequency band
                        self.select_band(0,1,self.mode_index)

                        # ME7TAG TEPLOT EL BEFORE KAMAN
                    elif file_path.endswith('.wav'):
                        # Process WAV file
                        self.before_first_time = True
                        self.after_first_time = True
                        self.plot_audio_waveform(file_path)
                        Fs = 1 / (self.x_values[1] - self.x_values[0])
                        self.Fs = np.ceil(Fs)
                        self.time_step = 1 / self.Fs
                        self.N = len(self.signal_data)
                        # Compute the FFT of the signal
                        self.fft()
                        # Select the frequency band
                        self.select_band(0, 1,self.mode_index)
                        ## Khaleeeeeeed
                        # self.audio_data, _ = sf.read(file_path)  # Read audio data
                        # self.load_audio(file_path)  # Load audio for playback
                        ############################
                        self.sample_rate, audio = scipy.io.wavfile.read(file_path)
                        self.new_audio_data = audio[:, 0] if len(audio.shape) > 1 else audio  # Consider only one channel if stereo
                        self.new_signal_data=self.new_audio_data
                        # print(len(self.signal_data))
                        # print(len(self.new_signal_data))
                        # Perform Fourier transform
                        # self.transformed_data = np.fft.fft(self.new_audio_data)
                        ###########################

                        self.create_spectrogram(self.signal_data, self.before_spectro_widget)
                        self.show_spectro_checkBox.setChecked(True)
                    else:
                        print("Unsupported file format.")

                except Exception as e:
                    print(f'Error reading file: {str(e)}')


    def fft(self):
        # Compute the FFT of the signal
        self.original_freq_signal_data = np.fft.rfft(self.signal_data)
        self.copy_freq_signal_data=np.copy(self.original_freq_signal_data)
        self.magnitudes=np.abs(self.original_freq_signal_data)
        self.phases=np.angle(self.original_freq_signal_data)
        self.original_frequency_components = np.fft.rfftfreq(self.N, self.time_step)
        # Define the frequency step and create the frequency values array
        frequency_step = self.Fs / self.N
        self.f_values = np.linspace(0, (self.N - 1) * frequency_step, self.N)
        self.f_values_plot = self.f_values[:int(self.N / 2 + 1)]

    def slider_value_changed(self, value, index):
        mode_index=self.mode_index
        print(value,index,mode_index)
        self.select_band(index,value, mode_index)
        self.after_spectro_widget.show()
        self.after_first_time = True
        self.create_spectrogram(self.y_modified, self.after_spectro_widget)
    def select_band(self,slider_index,value,mode_index,):
        print(mode_index)
        if mode_index==0: # Validation
            # Select the frequency band
            f_band_min,f_neg_band_min = (slider_index*10),(-slider_index*10)
            f_band_max,f_neg_band_max = ((slider_index+1)*10+1),(-((slider_index+1)*10+1))
        elif mode_index==1: # Music
            # Select the frequency band
            f_band_min, f_neg_band_min = (slider_index * 10000), (-slider_index * 10000)
            f_band_max, f_neg_band_max = ((slider_index + 1) * 10000 + 1), (-((slider_index + 1) * 10000 + 1))
        elif mode_index==2: # ECG
            f_band_min, f_neg_band_min = (slider_index * 10), (-slider_index * 10)
            f_band_max, f_neg_band_max = ((slider_index + 1) * 10 + 1), (-((slider_index + 1) * 10 + 1))
        else: # Animals
            if slider_index==0:
                f_band_min, f_neg_band_min = (600), (-600)
                f_band_max, f_neg_band_max = (800), (-800)
            elif slider_index==1:
                f_band_min, f_neg_band_min = (3800), (-3800)
                f_band_max, f_neg_band_max = (7000), (-7000)
            elif slider_index==2:
                f_band_min, f_neg_band_min = (1000), (-1000)
                f_band_max, f_neg_band_max = (25000), (-25000)
            else:
                f_band_min, f_neg_band_min = (0), (0)
                f_band_max, f_neg_band_max = (3000), (3000)
        band_pos_indices = np.where((self.original_frequency_components >= f_band_min) & (self.original_frequency_components <= f_band_max))[0]
        band_neg_indices = np.where((self.original_frequency_components <= f_neg_band_min) & (self.original_frequency_components >= f_neg_band_max))[0]
        band_indices = np.concatenate((band_pos_indices, band_neg_indices))
        self.multiply_band_by_window(window_index=global_variable,band=band_indices,value=value)

    def multiply_band_by_window(self, window_index, band, value):
        copy_band = np.copy(self.original_freq_signal_data[band])
        # Extract magnitude and phase
        copy_band_mag = np.abs(copy_band)
        copy_band_phase = np.angle(copy_band)
        if window_index == 0:  # Gaussian waveform
            # Multiply the selected frequency band by a Gaussian window on the magnitude
            band_width = len(band)
            sigma = 20  # Adjust the standard deviation as needed
            gaussian_window_manual = np.exp(-((np.arange(band_width) - (band_width - 1) / 2) ** 2) / (2 * sigma ** 2))
            copy_band_mag *= (value * gaussian_window_manual)
        elif window_index == 1:  # Rectangle waveform
            # Multiply the selected frequency band by a rectangular window on the magnitude
            rectangular_window_manual = np.zeros_like(self.copy_freq_signal_data, dtype=float)
            rectangular_window_manual[band] = 1
            copy_band_mag *= (value * rectangular_window_manual[band])
        elif window_index == 2:  # Hamming waveform
            # Multiply the selected frequency band by a Hamming window on the magnitude
            band_width = len(band)
            hamming_window_manual = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(band_width) / (band_width - 1))
            copy_band_mag *= (value * hamming_window_manual)
        else:  # Hanning waveform
            # Multiply the selected frequency band by a Hanning window on the magnitude
            band_width = len(band)
            hanning_window_manual = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(band_width) / (band_width - 1))
            copy_band_mag *= (value * hanning_window_manual)
        # Combine magnitude and phase back into complex values
        copy_band = copy_band_mag * np.exp(1j * copy_band_phase)
        self.plot_signal_freq_domain(self.copy_freq_signal_data, copy_band, band)
    def plot_signal_freq_domain(self, copy_data, band_data, band):
        # Clear the current graph
        self.frequency_widget.clear()
        # Combine the modified band back into the data
        copy_data[band] = band_data
        # Compute the magnitude spectrum for plotting
        freq_signal_data_mag = np.abs(copy_data) / self.N
        freq_signal_data_mag_plot = 2 * freq_signal_data_mag[:int(self.N / 2) + 1]
        freq_signal_data_mag_plot[0] /= 2
        self.frequency_widget.plot(self.f_values_plot, freq_signal_data_mag_plot)
        self.ifft(copy_data)

    def ifft(self,copy_data):
        # Apply inverse FFT
        y_modified = np.fft.irfft(copy_data).real
        self.inv_transform=y_modified
        self.y_modified = y_modified
        # Plot modified signal in time domain
        self.after_time_widget.clear()
        self.after_time_widget.plot(self.x_values, self.y_modified, pen='r', name='Modified Signal')
    def load_audio(self, file_path):
        # Load audio file for playback
        self.media_player_before.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

    def plot_audio_waveform(self, file_path):
        wav_file = wave.open(file_path, 'rb')
        # Get audio data
        num_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        # Calculate time values
        self.frame_rate = wav_file.getframerate()
        duration = num_frames / self.frame_rate
        self.x_max_range=duration
        time_values = np.linspace(0, duration, len(audio_data))
        self.signal_data=audio_data
        # self.signal_data=self.signal_data[:285436]
        self.x_values=time_values
        # Plot the audio waveform
        self.before_time_widget.plot(time_values, audio_data, pen='b')
        self.before_time_widget.setLabel('left', 'Amplitude')
        self.before_time_widget.setLabel('bottom', 'Time', units='s')

    def plot_from_csv(self, x_axis, y_axis, widget):
        widget.clear()
        widget.plot(x_axis, y_axis, pen='b')

    def create_spectrogram(self, signal, widget):
        if hasattr(widget, 'canvas') and widget.canvas:
            widget.canvas.setParent(None)
            widget.fig.clear()
        fig, ax = plt.subplots()
        Fs = self.Fs
        spec = ax.specgram(signal, Fs=Fs, cmap='plasma')
        # Add color bar
        cbar = fig.colorbar(spec[3], ax=ax)
        cbar.set_label('Intensity (dB)')
        widget.canvas = FigureCanvas(fig)
        widget_layout = widget.layout()
        widget_layout.addWidget(widget.canvas)
        widget.fig = fig
        widget.canvas.draw()

    def show_hide_spectrogram(self, state, widget):
        if widget.fig is not None:
            if state == Qt.Checked:
                widget.fig.set_visible(True)
                widget.show()
                if widget.canvas not in widget.findChildren(QWidget):
                    widget_layout = widget.layout()
                    widget_layout.addWidget(widget.canvas)
            else:
                widget.fig.set_visible(False)
                if widget.canvas in widget.findChildren(QWidget):
                    widget_layout = widget.layout()
                    widget_layout.removeWidget(widget.canvas)
                    widget.hide()
            widget.canvas.draw()

    def handle_spectro_check_box(self, is_checked):
        index = self.before_or_after.currentIndex()
        if index == 1:
            self.show_hide_spectrogram(is_checked, self.before_spectro_widget)
        elif index == 2:
            self.show_hide_spectrogram(is_checked, self.after_spectro_widget)

        elif index == 0:
            self.show_hide_spectrogram(is_checked, self.before_spectro_widget)
            self.show_hide_spectrogram(is_checked, self.after_spectro_widget)


class PopupDialog(QDialog, ui_popup):
    def __init__(self):
        QDialog.__init__(self)
        self.setupUi(self)

        #DECLARING AND CONNECTING THE SMOOTHED WINDOW GRAPH LAYOUT
        self.popup_widget = pg.PlotWidget()
        window_graph_layout = self.findChild(QHBoxLayout, 'window_graph_layout')
        window_graph_layout.addWidget(self.popup_widget)
        #CONNECTING THE SMOOTHER COMBOBOXES
        self.smoother_combobox=self.findChild(QComboBox,'smoother_combobox')
        self.smoother_combobox.setCurrentIndex(self.smoother_combobox.findText("Gaussian"))
        self.smoother_combobox.currentIndexChanged.connect(self.updateSmoothedGraph)
        self.smoother_combobox.currentIndexChanged.connect(self.show_hide_parameters)

        #ADDING LINE EDITS AND LABELS
        self.smoother_line_edit_3 = self.findChild(QLineEdit, 'smoother_line_edit_3')
        self.smoother_label_3 = self.findChild(QLabel, 'smoother_label_3')

        #CONNECTING SMOOTHER BUTTONS
        self.smoother_plot_btn = self.findChild(QPushButton, 'smoother_plot_btn')
        self.smoother_plot_btn.clicked.connect(self.get_parameters)
        self.smoother_done_btn = self.findChild(QPushButton, 'smoother_done_btn')
        self.smoother_done_btn.clicked.connect(self.done_btn_clicked)
        self.smoother_done_btn.clicked.connect(self.accept)

    def updateSmoothedGraph(self, index, sigma=0.2, f=1.0, mu=0.0):
        start = -0.0001
        end = 1
        t = np.linspace(start, end, 1000)
        if index == 0:  # Gaussian waveform
            start = mu - 1
            end = mu + 1
            t = np.linspace(start, end, 1000)
            waveform = self.gaussian_wave(t, mu, sigma)
        elif index == 1:  # Rectangle waveform
            waveform = self.rectangular_wave(t, f)
        elif index == 2:  # Hamming waveform
            waveform = self.hamming_wave(t)
        else:  # Hanning waveform
            waveform = self.hanning_wave(t)
        # Clear the current plot and plot the selected waveform
        curve = pg.PlotDataItem(t, waveform, pen=pg.mkPen(color='r', width=2, style=Qt.SolidLine, antialias=True))
        self.popup_widget.clear()  # Clear existing items in the PlotWidget
        self.popup_widget.addItem(curve)  # Add the new curve

    # DECLARING THE FORMULAS OF DIFFERENT SMOOTHED WINDOW WAVES
    def gaussian_wave(self, t, mu, sigma):
        y = 1 / (sigma * (2 * np.pi) ** .5) * np.exp(-(t - mu) ** 2 / (2 * sigma ** 2))
        return y

    def rectangular_wave(self, t, f):
        omega = 2 * np.pi * f
        y = 0.5 * (1 + np.sign(np.sin(omega * t)))
        return y

    def hamming_wave(self, t):
        omega = 2 * np.pi * 1
        y = (0.54 - 0.46 * np.cos(omega * t))
        return y

    def hanning_wave(self, t):
        omega = 2 * np.pi * 1
        y = 0.5 * (1 - np.cos(omega * t))
        return y

    def get_parameters(self):
        try:
            sigma = float(self.smoother_line_edit_3.text())
        except ValueError:
            sigma = 1
        self.updateSmoothedGraph(self.smoother_combobox.currentIndex(), sigma)

    def show_hide_parameters(self, index):
        if index == 0:  # Gaussian waveform
            self.smoother_label_3.show()
            self.smoother_line_edit_3.show()
            self.smoother_line_edit_3.setText("Sigma")
        else:  # Not Gaussian
            self.smoother_label_3.hide()
            self.smoother_line_edit_3.hide()

    def done_btn_clicked(self):
        global global_variable
        global_variable = self.smoother_combobox.currentIndex()
        print(global_variable)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()