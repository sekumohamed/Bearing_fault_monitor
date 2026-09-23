"""
real_sensor_stream.py

Replays REAL held-out test-set windows (X_test.npy / y_test.npy) over the
/ws/sensor WebSocket, one full 1024-sample window per message, so the live
dashboard shows genuine model performance instead of synthetic signals the
model was never trained to recognize.

This is the correct way to demo "live" inference without physical hardware:
it proves the buffering -> windowing -> normalization -> inference ->
broadcast pipeline works end-to-end, using real signal the model actually
understands.

Run:
    python3 real_sensor_stream.py --ws ws://localhost:8000/ws/sensor
"""

import asyncio
import argparse
import json
import time
import numpy as np
import websockets

CLASS_NAMES = ['Normal', 'IR_007', 'IR_014', 'IR_021', 'Ball_007',
               'Ball_014', 'Ball_021', 'OR_007', 'OR_014', 'OR_021']


def load_test_set(x_path, y_path):
    X = np.load(x_path)
    y_raw = np.load(y_path, allow_pickle=True)

    # Normalize label format: could be int indices or string labels
    labels = []
    for val in y_raw:
        if isinstance(val, bytes):
            val = val.decode()
        if isinstance(val, str):
            labels.append(val)
        else:
            labels.append(CLASS_NAMES[int(val)])
    return X, labels


async def stream(ws_url, x_path, y_path, delay, shuffle, only_class, limit):
    X, labels = load_test_set(x_path, y_path)
    indices = list(range(len(X)))

    if only_class:
        indices = [i for i in indices if labels[i] == only_class]
        if not indices:
            raise SystemExit(f"No windows found for class '{only_class}'")

    if shuffle:
        np.random.shuffle(indices)

    if limit:
        indices = indices[:limit]

    print(f"Loaded {len(X)} test windows. Streaming {len(indices)} of them.")

    async with websockets.connect(ws_url) as ws:
        print(f"Connected to {ws_url}")
        for count, idx in enumerate(indices, 1):
            window = X[idx].astype(np.float32).flatten().tolist()
            true_label = labels[idx]

            # Send the full window as ONE message so it maps to exactly one
            # inference frame on the backend (no cross-window mixing).
            await ws.send(json.dumps({"samples": window, "true_label": true_label}))
            print(f"[{count}/{len(indices)}] sent window (true label: {true_label})")

            await asyncio.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Replay real CWRU test windows to the live inference backend.")
    parser.add_argument("--ws", default="ws://localhost:8000/ws/sensor", help="WebSocket URL of /ws/sensor endpoint")
    parser.add_argument("--x", default="X_test.npy", help="Path to X_test.npy")
    parser.add_argument("--y", default="y_test.npy", help="Path to y_test.npy")
    parser.add_argument("--delay", type=float, default=1.4, help="Seconds between windows (matches old sim pacing)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle window order instead of sequential")
    parser.add_argument("--class", dest="only_class", default=None, choices=CLASS_NAMES,
                         help="Only stream windows of this one class")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N windows")
    args = parser.parse_args()

    asyncio.run(stream(args.ws, args.x, args.y, args.delay, args.shuffle, args.only_class, args.limit))


if __name__ == "__main__":
    main()