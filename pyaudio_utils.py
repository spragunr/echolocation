"""
Audio recorder and player classes built on pyaudio.
"""
import time
import wave
import cStringIO
import pyaudio
import scipy.io.wavfile

class AudioPlayer(object):
    """ Asynchronous audio player. NOT WORKING WELL."""
    
    def __init__(self, wav_filename):
        self.wave_file = wave.open(wav_filename, 'rb')
        print self.wave_file.getnframes()
        self.pyaudio = pyaudio.PyAudio()
        self.out_stream = None
        self.pyaudio = pyaudio.PyAudio()

        
        self.out_stream = self.pyaudio.open(
            format=self.pyaudio.get_format_from_width(
                self.wave_file.getsampwidth()),
            channels=self.wave_file.getnchannels(),
            rate=self.wave_file.getframerate(),
            output=True,
            frames_per_buffer=256,
            stream_callback=self._callback)

        self.frame_bytes = (self.wave_file.getsampwidth() *
                            self.wave_file.getnchannels())
        self.playing = False
        self.out_stream.start_stream()

    def play(self):
        self.wave_file.rewind()
        self.playing = True

    def is_playing(self):
        return self.playing

    def shutdown(self):
        self.out_stream.stop_stream()
        self.out_stream.close()
        self.pyaudio.terminate()

    def _callback(self, in_data, frame_count, time_info, status):
        """ This is a pyAudio callback, not a ROS callback."""
        if self.playing is False:
            data = '\x00' * self.wave_file.getsampwidth() * frame_count
        else:
            data = self.wave_file.readframes(frame_count)
            frames_read = len(data) / self.frame_bytes
            if frame_count != frames_read:
                self.playing = False
                data = data + '\x00' * (self.wave_file.getsampwidth() *
                                        frame_count - frames_read)
        return (data, pyaudio.paContinue)


class AudioRecorder(object):
    """Asynchronous audio recorder. 
    
    Some code borrowed from: https://gist.github.com/sloria/5693955

    """

    def __init__(self, format=pyaudio.paInt16,
                 channels=1,
                 rate=44100, frames_per_buffer=256):
        self.pyaudio = pyaudio.PyAudio()
        self.channels = channels
        self.format = format
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

        self.stream = None
        self.done = False
        self.wavefile = None

    def _init_wav_file(self):
        """ Create a file-like object in memory to back up a wav file object
        Nothing is actually saved to disk."""
        self.stringio = cStringIO.StringIO()
        self.wavefile = wave.open(self.stringio, 'wb')
        self.wavefile.setnchannels(self.channels)
        self.wavefile.setsampwidth(
            self.pyaudio.get_sample_size(pyaudio.paInt16))
        self.wavefile.setframerate(self.rate)

    def record(self, duration):
        """ Start recording.  Continue recording for duration seconds. """
        
        self.done = False
        self._init_wav_file()
        self.target_callbacks = int(self.rate /
                                    self.frames_per_buffer * duration) + 2
        self.num_callbacks = 0

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.pyaudio.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.callback)
        self.stream.start_stream()


    def done_recording(self):
        return self.done

    def get_data(self):
        """Returns a tuple where the first entry is the sample rate and the
        second entry is a numpy array containing the audio samples.
        """
        
        if not self.done_recording():
            raise RuntimeError("Audio data not ready!")
        self.stringio.seek(0)
        result = scipy.io.wavfile.read(self.stringio)
        self.stream.stop_stream()
        self.stream.close()
        return result

    def shutdown(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        if self.num_callbacks > 1: # discard the first two (bad data)
            self.wavefile.writeframes(in_data)
            
        if self.num_callbacks < self.target_callbacks:
            done = pyaudio.paContinue
        else:
            done = pyaudio.paComplete

        self.num_callbacks += 1
        
        if self.num_callbacks < self.target_callbacks:
            return in_data, pyaudio.paContinue
        else:
            self.done = True
            return in_data, pyaudio.paComplete




if __name__ == "__main__":
    player = AudioPlayer('data/16000to8000.02s.wav')
    for i in range(10):
        player.play()
        time.sleep(.2)

    recorder = AudioRecorder()
    recorder.record(2)
    
    while player.is_playing() or not recorder.done_recording():
        recorder.record(.04)
        time.sleep(.05)

    print recorder.get_data()
    
