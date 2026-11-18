"""Serve the offline UI without a reverse-DNS lookup during server startup."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer


class UIServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer calls getfqdn(0.0.0.0), which can stall on offline hosts
        # and hosted runners. Static files need only the bound address.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / "ui"
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with UIServer((args.host, args.port), handler) as server:
        print(f"PathSnap UI listening on {args.host}:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
