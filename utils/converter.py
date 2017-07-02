"""Utility module for file conversion."""
import os

import numpy as np
import scipy.io.wavfile as wav
from pipes import quote


def mp3_to_wav(filepath, out_dir='dataset/binary', sample_rate=44100):
    """
    Converts an MP3 file to WAV format.
    :param filepath: The input file.
    :param out_dir: The output file path.
    :param sample_rate: The sampling rate of the audio file, which is converted internally.
    :return: The filepath of the converted WAV file.
    """
    if not filepath.endswith('.mp3'):
        raise "File not in MP3 format."

    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    tmp_file = os.path.join('/tmp', basename + '.mp3')
    out_filepath = os.path.join(out_dir, basename + '.wav')
    sample_rate_arg = '{0:.1f}'.format(float(sample_rate))

    cmd = 'lame -a -m m {0} {1}'.format(quote(filepath), quote(tmp_file))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(tmp_file), quote(out_filepath), sample_rate_arg)
    os.system(cmd)

    os.remove(tmp_file)
    return out_filepath


def read_from_wav(filepath):
    """
    Reads a numpy array from a WAV file.
    :param filepath: The file to read.
    :return: Returns a tuple (audio data, sample rate).
    """
    data = wav.read(filepath)
    ndarray = data[1].astype(np.float32) / 32767.0  # normalize to [-1, 1]
    return ndarray, data[0]


def write_as_wav(ndarray, filepath, sample_rate):
    """
    Writes the numpy array to a WAV file.
    :param ndarray: The audio data.
    :param filepath: The path to write the audio file.
    :param sample_rate: The audio sampling rate.
    """
    data = ndarray * 32767.0  # unnormalize
    data = data.astype(np.int16)
    wav.write(filepath, sample_rate, data)
