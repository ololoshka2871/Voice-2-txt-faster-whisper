#!/usr/bin/env python

import io
import logging
from aiohttp import web
import functools
import argparse

from transformers import AutoModelForCTC, Wav2Vec2Processor, AutoModelForSeq2SeqLM, T5TokenizerFast

import librosa
import torch
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(
            batch, device="cuda").unsqueeze(0)  # , device="cuda"
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch = processor.batch_decode(pred_ids)[0]
    return batch


async def index(request: web.Request) -> web.Response:
    # show api documentation
    return web.FileResponse('index.html')


async def recognize_post(models: dict, request: web.Request) -> web.StreamResponse:
    # get voice_id value
    # voice_id = request.rel_url.query.get('voice_id', None)

    # get text to say
    # text = (await request.read()).decode('utf-8')

    # logger.info(f'Generating audio ({voice_id}) for text: "{text}"')

    # generate audio file
    # data = tts.say(text, voice_id)

    # response = web.StreamResponse()
    # response.headers['Content-Type'] = 'audio/wav'

    # writer = await response.prepare(request)
    # await writer.write(data)
    # await writer.drain()

    # return response
    return None


async def start_server() -> web.Application:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = dict(
        wav2vec2_model=AutoModelForCTC.from_pretrained(
            "UrukHan/wav2vec2-russian", cache_dir="models").to(device),
        wav2vec2_processor=Wav2Vec2Processor.from_pretrained(
            "UrukHan/wav2vec2-russian", cache_dir="models", device=device),

        t5_ru_spell_model=AutoModelForSeq2SeqLM.from_pretrained(
            "UrukHan/t5-russian-spell", cache_dir="models").to(device),
        t5_ru_spell_tokenizer=T5TokenizerFast.from_pretrained(
            "UrukHan/t5-russian-spell", cache_dir="models", device=device),
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
        description='Use Seliro TTS to generate audio files, check models at: "https://models.silero.ai/models/tts/{{language}}/{{voice_model}}"'
    )
    parser.add_argument('-p', '--port', type=int,
                        default=3154, help='Port to listen on')
    args = parser.parse_args()

    logger.info(f'Starting server at http://localhost:{args.port}/')

    web.run_app(start_server(), port=args.port)
