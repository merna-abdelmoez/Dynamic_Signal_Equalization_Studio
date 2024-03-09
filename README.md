# Dynamic_Signal_Equalization_Studio

## Overview:
The Signal Equalizer is a desktop application designed to manipulate the magnitude of frequency components in a signal. It serves as a fundamental tool in the music and speech industry, as well as in biomedical applications such as hearing aid abnormalities detection.

## Features:

### Uniform Range Mode:

In this mode, the total frequency range of the input signal is divided into 10 equal ranges. Each range is controlled by a slider in the user interface.

### Musical Instruments Mode:

Users can fine-tune the magnitude of specific musical instruments within the input music signal. The signal is a blend of at least four different musical instruments.

### Animal Sounds Mode:

Similar to the Musical Instruments Mode, this mode allows users to adjust the magnitude of specific animal sounds within a mixture of at least four animal sounds.

### ECG Abnormalities Mode:

This mode enables users to manipulate the magnitude of arrhythmia components within the input signal. The signal comprises four ECG signals, including one normal signal and three signals with specific types of arrhythmia.

The application also offers the flexibility to apply a multiplication/smoothing window to the frequency range multiplied by the corresponding slider value. Users can choose from four windows (Rectangle, Hamming, Hanning, Gaussian), with customizable parameters.

## Libraries Used:

- **Python**: Programming language used for development.
- **PyQt5**: For building the desktop application GUI.
- **NumPy**: For numerical operations and array manipulations.
- **SciPy**: For scientific computing functions, including signal processing.
- **Matplotlib**: For creating visualizations, including spectrogram display.
- **SoundFile**: For reading and writing audio files.

## Preview:

![Screenshot 1](Task%203/ss/1.png)
![Screenshot 2](Task%203/ss/2.png)
![Screenshot 3](Task%203/ss/3.png)
![Screenshot 4](Task%203/ss/4.png)

## Contributors <a name = "Contributors"></a>

<table>
  <tr>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/OmarEmad101">
          <img src="https://github.com/OmarEmad101.png" width="100px" alt="@OmarEmad101">
          <br>
          <sub><b>Omar Emad</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/Omarnbl">
          <img src="https://github.com/Omarnbl.png" width="100px" alt="@Omarnbl">
          <br>
          <sub><b>Omar Nabil</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/KhaledBadr07">
          <img src="https://github.com/KhaledBadr07.png" width="100px" alt="@KhaledBadr07">
          <br>
          <sub><b>Khaled Badr</b></sub>
        </a>
      </div>
    </td>
    <td align="center">
      <div style="text-align:center; margin-right:20px;">
        <a href="https://github.com/merna-abdelmoez">
          <img src="https://github.com/merna-abdelmoez.png" width="100px" alt="@merna-abdelmoez">
          <br>
          <sub><b>Mirna Abdelmoez</b></sub>
        </a>
      </div>
    </td>
  </tr>
</table>

## Acknowledgments

**This project was supervised by Dr. Tamer Basha & Eng. Abdallah Darwish, who provided invaluable guidance and expertise throughout its development as a part of the Digital Signal Processing course at Cairo University Faculty of Engineering.**

<div style="text-align: right">
    <img src="https://imgur.com/Wk4nR0m.png" alt="Cairo University Logo" width="100" style="border-radius: 50%;"/>
</div>
