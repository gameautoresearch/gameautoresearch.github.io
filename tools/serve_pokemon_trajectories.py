"""Loopback-only, allowlisted derived-report server. Never serves evidence trees."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]/'reports/pokemon'
PREFIX='four-researcher-trajectories-2026-11-10'
ALLOWED={PREFIX+s for s in ['.html','.json','-continuation.html','-continuation.json','-audit.json','.png','-continuation.png']}
ALLOWED.update({'sol-kimi-research-behavior-2026-11-10'+s for s in ['.html','.json','.png']})
ALLOWED.add('four-model-4h-pilot-2026-11-09.md')
ALLOWED.update({'pairwise-pilot-2026-11-10'+s for s in ('.html','.json','.md','.png')})
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        name=urlsplit(self.path).path.lstrip('/')
        if not name:
            self.send_response(302);self.send_header('Location','/'+PREFIX+'.html');self.end_headers();return
        if name not in ALLOWED:
            self.send_error(404);return
        self.path='/'+name
        super().do_GET()
    def do_HEAD(self):
        if urlsplit(self.path).path.lstrip('/') not in ALLOWED:self.send_error(404);return
        super().do_HEAD()
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765);args=parser.parse_args()
    print(f'http://127.0.0.1:{args.port}/{PREFIX}.html',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,directory=str(ROOT))).serve_forever()
