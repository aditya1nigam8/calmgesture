"""Local webcam-to-sound web app. Run: python src/app.py (then localhost:8765)."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from live_engine import LivePredictor

ROOT = Path(__file__).resolve().parents[1]
STATIC = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
          '/style.css': ('style.css', 'text/css')}


class Runtime:
    def __init__(self, checkpoint, landmark_model):
        self.predictor = LivePredictor(checkpoint)
        self.landmark_model = landmark_model
        self.detector = None
        self.session = None
        self.last_frame = 0

    def start(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.stop()
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.landmark_model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.session = secrets.token_urlsafe(24)
        self.last_frame = time.monotonic()
        return self.session

    def stop(self):
        if self.detector is not None:
            self.detector.close()
        self.detector = None
        self.session = None
        self.predictor.reset()

    def frame(self, content):
        import mediapipe as mp
        started = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > 1920:
            raise ValueError('Invalid camera frame')
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = round(time.monotonic() * 1000)
        result = self.detector.detect_for_video(image, timestamp)
        detection_ms = (time.perf_counter() - started) * 1000
        found = bool(result.hand_landmarks)
        raw = np.zeros((21, 3), dtype=np.float32)
        points, handedness = [], None
        if found:
            raw = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
            points = raw[:, :2].tolist()
            raw[:, [0, 2]] *= image.width / image.height
            handedness = result.handedness[0][0].category_name
        response = self.predictor.update(raw, found, timestamp, handedness)
        response.update(landmarks=points, detection_ms=detection_ms,
                        processing_ms=(time.perf_counter() - started) * 1000)
        self.last_frame = time.monotonic()
        return response


def handler_for(runtime):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, status, data, content_type='application/json'):
            body = json.dumps(data).encode() if content_type == 'application/json' else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Permissions-Policy', 'camera=(self), microphone=()')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/api/status':
                return self.reply(200, {'model': '1D-CNN', 'classes': runtime.predictor.classes,
                                       'window_seconds': 3, 'sample_fps': 12, 'ready': True})
            if path not in STATIC:
                return self.reply(404, {'error': 'Not found'})
            filename, kind = STATIC[path]
            return self.reply(200, (ROOT / 'web' / filename).read_bytes(), kind)

        def do_POST(self):
            origin = self.headers.get('Origin')
            host = self.headers.get('Host')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if host not in allowed or (origin and origin not in {f'http://{host}'}):
                return self.reply(403, {'error': 'Use the local app address'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 1024 * 1024:
                    return self.reply(413, {'error': 'Frame too large'})
                content = self.rfile.read(length)
                if self.path == '/api/start':
                    if runtime.session and time.monotonic() - runtime.last_frame < 15:
                        return self.reply(409, {'error': 'A session is active. Stop it in the other tab first.'})
                    return self.reply(200, {'session': runtime.start()})
                if not runtime.session or not secrets.compare_digest(self.headers.get('X-Session', ''), runtime.session):
                    return self.reply(409, {'error': 'Session ended. Press Start again.'})
                if self.path == '/api/stop':
                    runtime.stop()
                    return self.reply(200, {'stopped': True})
                if self.path == '/api/frame':
                    if not content.startswith(b'\xff\xd8'):
                        raise ValueError('Expected a JPEG camera frame')
                    return self.reply(200, runtime.frame(content))
                return self.reply(404, {'error': 'Not found'})
            except ValueError as error:
                return self.reply(400, {'error': str(error)})
            except Exception:
                runtime.stop()
                self.log_error('Frame processing failed; see terminal traceback')
                import traceback
                traceback.print_exc()
                return self.reply(500, {'error': 'Tracking could not start or continue. Check the terminal, then restart.'})

        def log_message(self, format, *args):
            if '/api/frame' not in str(args):
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/jester_v1/1dcnn/model.pt')
    parser.add_argument('--landmark_model', type=Path, default=ROOT / 'data/models/hand_landmarker.task')
    args = parser.parse_args()
    for path in (args.checkpoint, args.landmark_model):
        if not path.is_file():
            parser.error(f'Missing {path}. See README setup and training instructions.')
    runtime = Runtime(args.checkpoint, args.landmark_model)
    server = HTTPServer(('127.0.0.1', args.port), handler_for(runtime))
    server.timeout = 1
    print(f'CalmGesture is ready at http://localhost:{args.port}. Press Ctrl+C to stop.', flush=True)
    try:
        while True:
            server.handle_request()
            if runtime.session and time.monotonic() - runtime.last_frame > 15:
                runtime.stop()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()
        server.server_close()


if __name__ == '__main__':
    main()
