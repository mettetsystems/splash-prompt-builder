"""Real-server smoke test: python tests/test_latency.py [--url ws://.../ws]."""
import argparse
import asyncio
import json
import statistics
import time

import websockets


async def measure_latency(url: str, iterations: int = 25, threshold_ms: float = 100) -> dict:
    if iterations < 1:
        raise ValueError('iterations must be at least 1')
    latencies = []
    async with websockets.connect(url, open_timeout=5) as ws:
        for i in range(iterations):
            text = f'a quiet lake at sunrise {i}'
            message = {'type': 'prompt_update', 'request_id': i, 'text': text, 'cursor': len(text)}
            started = time.perf_counter()
            await ws.send(json.dumps(message))
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            latency = (time.perf_counter() - started) * 1000
            if response.get('type') != 'diffusion_update' or response.get('request_id') != i:
                raise AssertionError(f'Unexpected response: {response}')
            anchors = ''.join(s['text'] for s in response['segments'] if s['kind'] == 'anchor')
            if anchors != text:
                raise AssertionError('Anchor fidelity failed')
            latencies.append(latency)
    p95 = sorted(latencies)[max(0, int(len(latencies) * .95 + .999) - 1)]
    return {'iterations': iterations, 'average_ms': statistics.mean(latencies),
            'min_ms': min(latencies), 'max_ms': max(latencies), 'p95_ms': p95,
            'threshold_ms': threshold_ms, 'passed': p95 < threshold_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='ws://127.0.0.1:8000/ws')
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--threshold-ms', type=float, default=100)
    args = parser.parse_args()
    try:
        result = asyncio.run(measure_latency(args.url, args.iterations, args.threshold_ms))
    except (OSError, TimeoutError, ValueError, AssertionError, websockets.exceptions.WebSocketException) as exc:
        print(f'ERROR: {exc}. Ensure the backend is running at {args.url}.')
        return 1
    print(json.dumps(result, indent=2))
    print('Measures mock WebSocket round trips; excludes browser debounce and is not model inference latency.')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
