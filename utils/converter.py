"""Utility module for file conversion."""
import os

import numpy as np
import scipy.io.wavfile as wav
from pipes import quote


def mp3_to_wav(filepath, sample_rate=44100, out_path='dataset/binary'):
    if not filepath.endswith('.mp3'):
        raise "File not in MP3 format."

    filename = os.path.basename(filepath)
    basename = os.path.splitext(filename)[0]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    tmp_file = os.path.join('/tmp', basename + '.mp3')
    out_name = os.path.join(out_path, basename + '.wav')
    sample_rate_arg = '{0:.1f}'.format(float(sample_rate))

    cmd = 'lame -a -m m {0} {1}'.format(quote(filepath), quote(tmp_file))
    os.system(cmd)
    cmd = 'lame --decode {0} {1} --resample {2}'.format(quote(tmp_file), quote(out_name), sample_rate_arg)
    os.system(cmd)

    os.remove(tmp_file)
    return out_name
