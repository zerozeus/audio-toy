import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import matplotlib.transforms
import os

default_image_width = 100
default_image_height = 100

default_fft_length = 1024
default_sample_rate = 44100

default_window_length = 25  #ms
default_stride_length = 15 #ms


class AudioParser(object):
    
    def __create_image__(self, data, sub_dir, name):
        dest_dir = os.path.join(self.out_dir, sub_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        file_name = os.path.join(dest_dir, name + '.jpg')
        if os.path.exists(file_name):
            return
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data, n_fft=default_fft_length, center=False)), ref=np.max)
        figure = plt.figure(figsize=(default_image_width, default_image_height), dpi=5)
        axis = plt.subplot(1, 1, 1)
        plt.axis('off')
        plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        librosa.display.specshow(D, x_axis='time', y_axis='linear')
        extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        extent = matplotlib.transforms.Bbox(extent.get_points()*np.array([[1.1],[.9]]))
        plt.savefig(file_name, format='jpg', bbox_inches=extent, pad_inches=0)
        plt.close()

    def __get_sample_data__(self, data, sample_rate, sample_content, trim = True):
        start_index = int(sample_content.start / 1000.0 * sample_rate)
        end_index = int(sample_content.end / 1000.0 * sample_rate)
        sample_data = data[start_index:end_index]
        if (trim):
            return librosa.effects.trim(sample_data)[0]
        return sample_data

    def __create_windowed_sample_image__(self, attribute, content, data, sample_rate, window = default_window_length, stride = default_stride_length):
        window = int(window / 1000.0 * sample_rate)
        stride = int(stride / 1000.0 * sample_rate)
        def windows(length, _window, _stride):
            start = 0
            while start + _window < length:
                yield start, start + _window
                start += _stride

        for (start, end) in windows(len(data), window, stride):
            name = self.__get_unique_name__(attribute, content, start, end)
            self.__create_image__(data[start: end], content.type.name, name)

    def __get_unique_name__(self, attribute, content, start, end):
        unique_name = []
        unique_name.append(os.path.splitext(attribute.file_name)[0])
        unique_name.append(str(content.start))
        unique_name.append(str(content.end))
        unique_name.append('['+str(start)+'-'+str(end)+']')
        return '-'.join(unique_name)

    def to_image(self, audio_sample, out_dir, trim = True):
        self.out_dir = out_dir
        for attribute, contents in audio_sample.content.iteritems():
            audio_path = os.path.join(audio_sample.audio_dir, attribute.sub_dir, attribute.file_name)
            data, sample_rate = librosa.load(audio_path, sr = default_sample_rate)
            for content in contents:
                try:
                    sample_data = self.__get_sample_data__(data, sample_rate, content, trim)
                    self.__create_windowed_sample_image__(attribute, content, sample_data, sample_rate)
                except:
                    print 'error attribute', attribute, 'content', content
        return out_dir


    def audio_to_image(self, audio, start, end, out_dir, trim = False, window = default_window_length, stride = default_stride_length):
        data, sample_rate = librosa.load(audio, sr = default_sample_rate)
        start_index = int(start / 1000.0 * sample_rate)
        end_index = int(end / 1000.0 * sample_rate)
        data = data[start_index:end_index]
        if (trim):
            data = librosa.effects.trim(data)[0]

        window = int(window / 1000.0 * sample_rate)
        stride = int(stride / 1000.0 * sample_rate)
        def windows(length, _window, _stride):
            _start = 0
            while _start + _window < length:
                yield _start, _start + _window
                _start += _stride
        audio_name = os.path.basename(audio).split('.')[0]
        self.out_dir = os.path.join(out_dir, audio_name)
        for (s, e) in windows(len(data), window, stride):
            unique_name = []
            unique_name.append("test")
            unique_name.append(audio_name)
            unique_name.append(str(start))
            unique_name.append(str(end))
            unique_name.append('['+str(s)+'-'+str(e)+']')
            self.__create_image__(data[s: e], '', '-'.join(unique_name))
        return self.out_dir