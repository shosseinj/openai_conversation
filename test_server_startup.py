import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class StartupTests(unittest.TestCase):
    def test_openai_only_import_and_startup_need_no_local_packages(self):
        code = '''
import asyncio, importlib.abc, sys
class NoLocal(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'numpy','torch','torchaudio','transformers','speechbrain','silero_vad','scipy','soundfile','soxr','live_vad','speaker_filter','noise_suppression','loudness_gate'}:
            raise AssertionError('Unexpected Local import: '+name)
sys.meta_path.insert(0, NoLocal())
import app
from fastapi import HTTPException
class Socket:
    async def accept(self): pass
    async def send_json(self, data): assert 'disabled' in data['error'].lower()
    async def close(self, code): assert code == 1008
async def check():
    async with app.lifespan(app.app):
        h = await app.health()
        assert h['local_enabled'] is False and h['default_engine'] == 'openai'
        assert not hasattr(app.app.state, 'transcriber')
        await app.listen(Socket())
        try: await app.enroll(None)
        except HTTPException as e: assert e.status_code == 503
        else: raise AssertionError('Local enrollment must be disabled')
asyncio.run(check())
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT,
                                env=dict(os.environ, ENABLE_LOCAL_STT='0'), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_port_cleanup_stops_existing_listener(self):
        from server_runtime import free_port
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        code = "import socket,time; s=socket.socket(); s.bind(('127.0.0.1',%d)); s.listen(); print('ready',flush=True); time.sleep(60)" % port
        child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            free_port(port)
            child.wait(timeout=3)
            with socket.socket() as s: s.bind(('127.0.0.1', port))
        finally:
            if child.poll() is None: child.kill(); child.wait()
            child.stdout.close()

    @unittest.skipUnless(os.name == 'posix', 'SIGTERM resistance is a Unix-specific case')
    def test_port_cleanup_force_kills_resistant_listener(self):
        from server_runtime import free_port
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        code = "import socket,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); s=socket.socket(); s.bind(('127.0.0.1',%d)); s.listen(); print('ready',flush=True); time.sleep(60)" % port
        child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            from contextlib import redirect_stdout
            import io
            output = io.StringIO()
            with redirect_stdout(output): free_port(port)
            self.assertIn('forcing termination', output.getvalue())
            # psutil.wait_procs can reap our own child before Popen.wait reads
            # its exit status; verify the contract: forced stop and released port.
            child.wait(timeout=3)
            with socket.socket() as s: s.bind(('127.0.0.1', port))
        finally:
            if child.poll() is None: child.kill(); child.wait()
            child.stdout.close()

    def test_missing_https_files_are_generated_and_existing_files_reused(self):
        from server_runtime import ensure_https
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            key, cert = ensure_https(root, names=['localhost', '127.0.0.1'])
            self.assertTrue(key.is_file()); self.assertTrue(cert.is_file())
            self.assertTrue((root / 'certs' / 'ca.crt').is_file())
            ctx = ssl.create_default_context(cafile=str(root / 'certs' / 'ca.crt'))
            ctx.load_cert_chain(str(cert), str(key))
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            import urllib.request
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
                def log_message(self, *args): pass
            server = HTTPServer(('127.0.0.1', 0), Handler)
            transport = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            transport.load_cert_chain(str(cert), str(key))
            server.socket = transport.wrap_socket(server.socket, server_side=True)
            worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
            try:
                with urllib.request.urlopen('https://localhost:'+str(server.server_port), context=ctx, timeout=3) as response:
                    self.assertEqual(response.read(), b'ok')
            finally:
                server.shutdown(); server.server_close(); worker.join(timeout=3)
            original = key.read_bytes(), cert.read_bytes()
            self.assertEqual(ensure_https(root), (key, cert))
            self.assertEqual(original, (key.read_bytes(), cert.read_bytes()))

    def test_partial_certificate_pair_is_not_overwritten(self):
        from server_runtime import ensure_https
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'certs').mkdir()
            key = root / 'certs' / 'server.key'; key.write_text('keep existing')
            with self.assertRaises(RuntimeError): ensure_https(root)
            self.assertEqual(key.read_text(), 'keep existing')


if __name__ == '__main__': unittest.main()
