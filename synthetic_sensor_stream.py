"""
synthetic_sensor_stream.py

Simulates a live vibration sensor by generating non-repeating synthetic
signal data and streaming it to the FastAPI backend's /ws/sensor endpoint,
exactly as a real ESP32 + ADXL345 would.

Use this to test the full live-inference pipeline (buffering, windowing,
normalization, model inference, dashboard broadcast) without hardware.

Run:
    pip install websockets numpy
    python3 synthetic_sensor_stream.py --ws ws://localhost:8000/ws/sensor

Requires app.py's /ws/sensor endpoint (from the buffering/inference stage)
to already be running.
"""

import asyncio
import argparse
import json
import time
import numpy as np
import websockets

# Rough characteristic-frequency multipliers per class, loosely inspired by
# CWRU fault-frequency relationships (BPFO/BPFI/BSF ratios to shaft speed).
# Not physically exact -- good enough to give each class a distinct signature
# for a live demo, not for publication-grade simulation.
CLASS_PROFILES = {
    "Normal":   {"base_freq": (25, 35),   "amp": (0.2, 0.5), "harmonics": [1.0]},
    "IR_007":   {"base_freq": (150, 170), "amp": (0.6, 0.9), "harmonics": [1.0, 2.0]},
    "IR_014":   {"base_freq": (150, 170), "amp": (0.9, 1.3), "harmonics": [1.0, 2.0, 3.0]},
    "IR_021":   {"base_freq": (150, 170), "amp": (1.3, 1.8), "harmonics": [1.0, 2.0, 3.0]},
    "Ball_007": {"base_freq": (110, 130), "amp": (0.6, 0.9), "harmonics": [1.0, 1.5]},
    "Ball_014": {"base_freq": (110, 130), "amp": (0.9, 1.3), "harmonics": [1.0, 1.5]},
    "Ball_021": {"base_freq": (110, 130), "amp": (1.3, 1.8), "harmonics": [1.0, 1.5]},
    "OR_007":   {"base_freq": (90, 105),  "amp": (0.6, 0.9), "harmonics": [1.0, 2.0]},
    "OR_014":   {"base_freq": (90, 105),  "amp": (0.9, 1.3), "harmonics": [1.0, 2.0]},
    "OR_021":   {"base_freq": (90, 105),  "amp": (1.3, 1.8), "harmonics": [1.0, 2.0]},
}
CLASS_NAMES = list(CLASS_PROFILES.keys())


def generate_batch(t0, dt, batch_size, profile, noise_std):
    """Generate `batch_size` non-repeating samples starting at time t0."""
    freq = np.random.uniform(*profile["base_freq"])       # re-randomized every batch
    amp = np.random.uniform(*profile["amp"])
    phase = np.random.uniform(0, 2 * np.pi)                # random phase offset
    t = t0 + np.arange(batch_size) * dt

    signal = np.zeros(batch_size)
    for h in profile["harmonics"]:
        harmonic_amp = amp / (h ** 0.5)                     # decaying harmonic amplitude
        signal += harmonic_amp * np.sin(2 * np.pi * freq * h * t + phase)

    noise = np.random.normal(0, noise_std, batch_size)       # fresh noise every call
    # slow random-walk drift so consecutive windows aren't just phase-shifted copies
    drift = np.cumsum(np.random.normal(0, 0.01, batch_size))

    return (signal + noise + drift).tolist(), t[-1] + dt


async def stream(ws_url, sample_rate_hz, batch_size, noise_std, class_switch_seconds, fixed_class):
    t = 0.0
    dt = 1.0 / sample_rate_hz
    current_class = fixed_class or np.random.choice(CLASS_NAMES)
    last_switch = time.time()

    async with websockets.connect(ws_url) as ws:
        print(f"Connected to {ws_url}. Streaming as class: {current_class}")
        while True:
            if not fixed_class and (time.time() - last_switch) > class_switch_seconds:
                current_class = np.random.choice(CLASS_NAMES)
                last_switch = time.time()
                print(f"[switch] now simulating: {current_class}")

            profile = CLASS_PROFILES[current_class]
            batch, t = generate_batch(t, dt, batch_size, profile, noise_std)

            # Send ground truth alongside the batch so the backend/dashboard
            # can validate live accuracy, same as the uploaded-file flow does.
            await ws.send(json.dumps({"samples": batch, "true_label": current_class}))

            await asyncio.sleep(batch_size / sample_rate_hz)


def main():
    parser = argparse.ArgumentParser(description="Stream synthetic vibration data to the live inference backend.")
    parser.add_argument("--ws", default="ws://localhost:8000/ws/sensor", help="WebSocket URL of /ws/sensor endpoint")
    parser.add_argument("--rate", type=int, default=800, help="Simulated sample rate in Hz")
    parser.add_argument("--batch", type=int, default=32, help="Samples per WebSocket message")
    parser.add_argument("--noise", type=float, default=0.15, help="Std dev of additive noise")
    parser.add_argument("--switch-every", type=float, default=15.0, help="Seconds between random class switches")
    parser.add_argument("--class", dest="fixed_class", default=None, choices=CLASS_NAMES,
                         help="Stick to one class instead of randomly switching")
    args = parser.parse_args()

    asyncio.run(stream(args.ws, args.rate, args.batch, args.noise, args.switch_every, args.fixed_class))


if __name__ == "__main__":
    main()