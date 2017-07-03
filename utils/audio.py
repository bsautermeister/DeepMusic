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
    audio = data[1].astype(np.float32) / 32767.0  # normalize to [-1, 1]
    return audio, data[0]


def write_as_wav(audio, filepath, sample_rate):
    """
    Writes the numpy array to a WAV file.
    :param ndarray: The audio data.
    :param filepath: The path to write the audio file.
    :param sample_rate: The audio sampling rate.
    """
    data = audio * 32767.0  # unnormalize
    data = data.astype(np.int16)
    wav.write(filepath, sample_rate, data)


def to_sample_blocks(audio, block_size):
    """
    Converts a numpy audio data array to a list of smaller blocks.
    :param audio: The audio data array.
    :param block_size: The block size.
    :return: A list of numpy arrays.
    """
    block_list = []
    total_samples = audio.shape[0]
    samples_counter = 0

    while samples_counter < total_samples:
        block = audio[samples_counter:samples_counter + block_size]

        if block.shape[0] < block_size:
            # pad audio with zeros
            padding = np.zeros((block_size - block.shape[0],))
            block = np.concatenate((block, padding))

        block_list.append(block)
        samples_counter += block_size
    return block_list


def from_sample_blocks(block_list):
    """
    Converts to an audio array given a list of samples blocks.
    :param block_list: The list of audio blocks.
    :return: The audio numpy array.
    """
    audio = np.concatenate(block_list)
    return audio


def to_fft_blocks(block_list):
    """
    Converts audio blocks in time domain to frequency domain.
    :param block_list: The audio blocks in time domain.
    :return: List of audio blocks in frequency domain.
    """
    fft_list = []
    for block in block_list:
        fft_block = np.fft.fft(block)
        new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_list.append(new_block)
    return fft_list


def from_fft_blocks(fft_list):
    """
    Converts audio blocks in frequency domain to time domain.
    :param fft_list: The audio blocks in frquency domain.
    :return:  List of audio blocks in time domain.
    """
    block_list = []
    for block in fft_list:
        num_elems = int(block.shape[0] / 2)
        real_elems = block[0:num_elems]
        imag_elems = block[num_elems:]
        block = np.fft.ifft(real_elems + 1.0j * imag_elems)
        block_list.append(block)
    return block_list
