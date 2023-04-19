#!/usr/bin/env python

import io
import logging
import functools
import argparse
import librosa
import torch

from aiohttp import web
from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, T5TokenizerFast


CACHE_DIR = "models"
MAX_INPUT = 256


logger = logging.getLogger(__name__)


def transcript(sound_data, model, processor, device) -> str:
    with torch.no_grad():
        input_values = torch.tensor(sound_data, device=device).unsqueeze(0)
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def correct_text(text, model, tokenizer, device) -> str:
    with torch.no_grad():
        input_ids = tokenizer([text], padding="longest",
                              max_length=MAX_INPUT,
                              truncation=True,
                              return_tensors="pt",).to(device)
        outputs = model.generate(**input_ids)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


async def index(request: web.Request) -> web.Response:
    # show api documentation
    return web.FileResponse('index.html')


async def recognize_post(config: dict, request: web.Request) -> web.StreamResponse:
    if request.headers["Content-Type"] != "audio/wav":
        return web.Response(status=415, text="Unsupported Input Media Type")

    wav_data = await request.read()

    audio, _sr = librosa.load(io.BytesIO(
        wav_data), sr=config['sample_rate'], mono=True)

    transcripted_text = transcript(audio, config['wav2vec2_model'],
                                   config['wav2vec2_processor'], device=config['device'])
    logger.info(f'Transcripted text: "{transcripted_text}"')

    corrected_text = correct_text(transcripted_text, config['t5_ru_spell_model'],
                                  config['t5_ru_spell_tokenizer'], device=config['device'])
    logger.info(f'Corrected text: "{corrected_text}"')

    # return transcripted_text and correct_text as json
    return web.json_response({'transcripted_text': transcripted_text,
                              'corrected_text': corrected_text})


async def start_server() -> web.Application:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        torch.set_num_threads(8)

    WAV2VEC2_RU = "UrukHan/wav2vec2-russian"
    T5_RU_SPELL = "UrukHan/t5-russian-spell"
    SAMPLE_RATE = 16000  # see model specifications

    logger.info(f'Loading AI models to {device}...')

    models = dict(
        wav2vec2_model=AutoModelForCTC.from_pretrained(
            WAV2VEC2_RU, cache_dir=CACHE_DIR).to(device),
        wav2vec2_processor=Wav2Vec2Processor.from_pretrained(
            WAV2VEC2_RU, cache_dir=CACHE_DIR, device=device),

        t5_ru_spell_model=AutoModelForSeq2SeqLM.from_pretrained(
            T5_RU_SPELL, cache_dir=CACHE_DIR).to(device),
        t5_ru_spell_tokenizer=T5TokenizerFast.from_pretrained(
            T5_RU_SPELL, cache_dir=CACHE_DIR, device=device),
        sample_rate=SAMPLE_RATE,
        device=device
    )

    app = web.Application()

    # call handle_request with tts as first argument
    app.add_routes([
        web.get('/', handler=index),
        web.post('/recognize', handler=functools.partial(recognize_post, models))
    ])
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='An AI voice to text transcription server')
    
    parser.add_argument('-p', '--port', type=int,
                        default=3154, help='Port to listen on')
    args = parser.parse_args()

    logger.info(f'Starting server at http://localhost:{args.port}/')

    web.run_app(start_server(), port=args.port)
