#!/usr/bin/env python3
"""Serve the read-only API on the loopback interface.

Binds 127.0.0.1 by design. There is no authentication and the payloads
describe private photographs, so the server must not be reachable from
the LAN. --host is deliberately not an option; exposing this needs a
reverse proxy that terminates auth, which is a decision to make on
purpose rather than by passing a flag.

Encoders are not loaded at startup. The first search loads them and the
server holds them from then on, which takes a search from 5.5s to under
a tenth of a second.
"""
import argparse
import sys

import uvicorn

HOST = "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on source changes. Development only.",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    print()
    print("=" * 76)
    print("VISUALMIND - API")
    print("=" * 76)
    print("Listening:  http://" + HOST + ":" + str(args.port))
    print("Docs:       http://" + HOST + ":" + str(args.port) + "/docs")
    print("Bound to loopback only - no auth, do not expose.")
    print()

    uvicorn.run(
        "visualmind.api:create_app",
        factory=True,
        host=HOST,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
