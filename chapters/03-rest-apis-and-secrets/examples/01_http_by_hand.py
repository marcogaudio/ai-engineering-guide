"""HTTP by hand: the exact text a client and a server exchange.

Starts a tiny HTTP server on this machine (nothing leaves your computer), then talks to it twice:
1. over a raw TCP socket, so you can read the literal request and response messages;
2. with httpx2, the HTTP client library that the openai and anthropic SDKs are built on.

Run: uv run chapters/03-rest-apis-and-secrets/examples/01_http_by_hand.py
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx2


class HelloHandler(BaseHTTPRequestHandler):
    """The server side: Python's standard library turns each request into a method call."""

    # Speak HTTP/1.1. The default, HTTP/1.0, closes the connection after every single response.
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # called for every GET request
        # `self.path` is the request target from the first line: path + query, e.g. "/hello?x=1".
        body = json.dumps({"message": "hello from the server", "path": self.path}).encode()
        self.send_response(200)  # the status line "HTTP/1.1 200 OK" (+ Server and Date headers)
        self.send_header("Content-Type", "application/json")  # what kind of data the body is
        self.send_header("Content-Length", str(len(body)))  # how many bytes the body has
        self.end_headers()  # the empty line that separates the headers from the body
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass  # silence the default "127.0.0.1 - - [date] GET /hello" line on stderr


def raw_http_get(host: str, port: int, path: str) -> str:
    """Do the four steps of the HTTP flow by hand and return the raw response text."""
    # Step 1: open a TCP connection to the server.
    with socket.create_connection((host, port), timeout=5) as sock:
        # Step 2: send an HTTP message. In HTTP/1.1 it is plain text: a request line
        # (method, path, version), one header per line, then an empty line. Lines end with
        # "\r\n" (carriage return + line feed), as the HTTP specification requires.
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"  # the only header HTTP/1.1 makes mandatory
            "Accept: application/json\r\n"  # the format we'd like back
            "Connection: close\r\n"  # ask the server to close the connection after replying
            "\r\n"
        )
        print("----- request sent -----")
        print(request.replace("\r\n", "\n"), end="")
        sock.sendall(request.encode())

        # Step 3: read the response. Bytes arrive in pieces, so keep reading until the server
        # closes the connection (recv() then returns b"", which ends the loop).
        chunks: list[bytes] = []
        while chunk := sock.recv(4096):
            chunks.append(chunk)
    # Step 4: leaving the `with` block closed our side of the connection.
    return b"".join(chunks).decode()


def with_httpx2(base_url: str) -> None:
    """The same request through a real HTTP client: it writes and parses the text for us."""
    # One Client = one pool of connections, reused for every request (HTTP/1.1 keep-alive).
    # Always set a timeout: without one, a stuck server could make your program hang forever.
    with httpx2.Client(base_url=base_url, timeout=5) as client:
        # `params` builds the query string: /hello?name=Ada&lang=en
        response = client.get("/hello", params={"name": "Ada", "lang": "en"})

    print("----- the same call with httpx2 -----")
    print("request line:  ", response.request.method, response.request.url)
    # httpx2 added these headers for us (Host, Accept, Accept-Encoding, Connection, User-Agent).
    print("request headers:", dict(response.request.headers))
    print("status:        ", response.http_version, response.status_code, response.reason_phrase)
    print("content-type:  ", response.headers["content-type"])  # header names are case-insensitive
    print("body as text:  ", response.text)
    print("body as JSON:  ", response.json())  # parsed into a Python dict


if __name__ == "__main__":
    # Port 0 tells the operating system "pick any free port", so this never clashes with
    # something else already running on your machine.
    server = ThreadingHTTPServer(("127.0.0.1", 0), HelloHandler)
    host, port = "127.0.0.1", server.server_port
    # Run the server in a background thread, so this same script can also be the client.
    # `daemon=True`: the thread won't keep the program alive once the main code is done.
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        raw = raw_http_get(host, port, "/hello")
        print("----- raw response received -----")
        print(raw.replace("\r\n", "\n"))
        print()
        with_httpx2(f"http://{host}:{port}")
    finally:
        server.shutdown()  # stop serve_forever()
        server.server_close()  # release the port
