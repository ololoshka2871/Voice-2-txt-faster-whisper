import io
import os
from typing import Optional
import torch
import wave


# https://github.com/snakers4/silero-models#text-to-speech
class SileroTTS:
    def __init__(self, language, model, model_file=None) -> None:
        self._localfile = f'{language}_{model}.pt' if model_file is None else model_file

        # download model if not exists
        if not os.path.isfile(self._localfile):
            torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{language}/{model}.pt',
                                           self._localfile)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(8)

        self._model = torch.package.PackageImporter(self._localfile).load_pickle(  # type: ignore
            "tts_models", "model")
        self._model.to(device)

    def say_to_file(self, text: str, speaker: str, outfile=None) -> str:
        '''
        Say text using speaker and save to file
        '''
        return self._model.save_wav(text=text,
                                    speaker=speaker,
                                    audio_path=outfile)

    def say(self, text: str, speaker: Optional[str] = None) -> bytes:
        '''
        Say text using speaker
        return (audio_data, sample_rate, sample_width)
        '''
        sample_rate = 48000

        if speaker is None:
            audiodata = self._model.apply_tts(text=text)
        else:
            audiodata = self._model.apply_tts(text=text, speaker=speaker)

        # create audio buffer from audio_data
        buffer = io.BytesIO()
        wf = wave.open(buffer, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audiodata * 32767).numpy().astype('int16'))
        wf.close()

        return buffer.getvalue()
