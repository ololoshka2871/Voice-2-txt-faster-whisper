#!/usr/bin/env python

import io
import logging
from aiohttp import web
import functools
import argparse


logger = logging.getLogger(__name__)


async def index(request: web.Request) -> web.Response:
    # show api documentation
    return web.FileResponse('index.html')


#async def say_get(tts: SileroTTS, request: web.Request) -> web.StreamResponse:
#    params = request.rel_url.query
#
#    # get voice_id value
#    voice_id = request.rel_url.query.get('voice_id', None)
#
#    # get text to say
#    text = params.get('text', '')
#
#    logger.info(f'Generating audio ({voice_id}) for text: "{text}"')
#
#    # generate audio file
#    data = tts.say(text, voice_id)
#
#    response = web.StreamResponse()
#    response.headers['Content-Type'] = 'audio/wav'
#
#    writer = await response.prepare(request)
#    await writer.write(data)
#    await writer.drain()
#
#    return response
#
#
#async def say_post(tts: SileroTTS, request: web.Request) -> web.StreamResponse:
#    # get voice_id value
#    voice_id = request.rel_url.query.get('voice_id', None)
#
#    # get text to say
#    text = (await request.read()).decode('utf-8')
#
#    logger.info(f'Generating audio ({voice_id}) for text: "{text}"')
#
#    # generate audio file
#    data = tts.say(text, voice_id)
#
#    response = web.StreamResponse()
#    response.headers['Content-Type'] = 'audio/wav'
#
#    writer = await response.prepare(request)
#    await writer.write(data)
#    await writer.drain()
#
#    return response
#

async def start_server() -> web.Application:
    app = web.Application()

    # call handle_request with tts as first argument
    app.add_routes([
        web.get('/', handler=index),
        #web.get('/say', handler=functools.partial(say_get, tts)),
        #web.post('/say', handler=functools.partial(say_post, tts))
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
