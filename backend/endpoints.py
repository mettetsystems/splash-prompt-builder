"""Read-only local server discovery; usable by setup before dependencies exist."""
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

DEFAULT_URLS = tuple(f'http://127.0.0.1:{port}/v1' for port in (11434, 1234, 8000, 8080, 5000))


def normalize_url(value):
    parsed = urlsplit(value.strip().rstrip('/'))
    host = parsed.hostname
    try:
        local = host == 'localhost' or ipaddress.ip_address(host or '').is_loopback
        parsed.port  # Validate the port even though the original authority is retained.
    except ValueError:
        local = False
    if (not local or parsed.scheme not in ('http', 'https') or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError('Use an HTTP(S) localhost or loopback endpoint without credentials, query, or fragment.')
    path = parsed.path.rstrip('/')
    if not path.endswith('/v1'):
        path += '/v1'
    return urlunsplit((parsed.scheme, parsed.netloc, path, '', ''))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def headers():
    token = os.environ.get('SPLASH_ENDPOINT_API_KEY', '')
    return {'Authorization': f'Bearer {token}'} if token else {}


def read_json(url, timeout=.6):
    # Local probes must not go through environment proxies or follow redirects.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, headers=headers()), timeout=timeout) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000:
        raise ValueError('Model listing is too large.')
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('Expected a model-list object.')
    return data


def probe_endpoint(url):
    base = normalize_url(url)
    root = base[:-3]
    result = {'base_url': base, 'provider': 'openai-compatible', 'models': [], 'loaded_only': False}
    # Native inventories distinguish loaded models from files on disk. Never
    # offer Ollama's /api/tags as a list of running models.
    for path, provider in (('/api/ps', 'ollama'), ('/api/v1/models', 'lmstudio')):
        try:
            data = read_json(root + path)
            items = data.get('models')
            if not isinstance(items, list):
                continue
            if provider == 'ollama':
                found = [{'id': m.get('name') or m.get('model'), 'context': m.get('context_length')}
                         for m in items if isinstance(m, dict)]
            else:
                # An unrelated /models route is not evidence of LM Studio.
                if items and not any(isinstance(m, dict) and 'loaded_instances' in m for m in items):
                    continue
                found = [{'id': instance.get('id'), 'context': instance.get('config', {}).get('context_length')}
                         for m in items if isinstance(m, dict) and m.get('type') == 'llm'
                         for instance in m.get('loaded_instances', []) if isinstance(instance, dict)]
            result.update(provider=provider, loaded_only=True,
                          models=[m for m in found if isinstance(m['id'], str) and m['id']])
            return result
        except HTTPError as exc:
            exc.close()
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    try:
        data = read_json(base + '/models')
        if not isinstance(data.get('data'), list):
            return None
        result['models'] = [{'id': m['id'], 'context': None} for m in data['data']
                            if isinstance(m, dict) and isinstance(m.get('id'), str) and m['id']]
        result['notice'] = 'This server does not report loaded state; it may load a model when used.'
        return result
    except HTTPError as exc:
        exc.close()
        if exc.code in (401, 403):
            result.update(auth_required=True, notice='Authentication required. Set SPLASH_ENDPOINT_API_KEY and rescan.')
            return result
    except (OSError, ValueError, TypeError):
        pass
    return None


def discover_endpoints(urls=None):
    configured = [u.strip() for u in os.environ.get('SPLASH_ENDPOINT_URLS', '').split(',') if u.strip()]
    candidates = list(dict.fromkeys(normalize_url(u) for u in (urls if urls is not None else (*DEFAULT_URLS, *configured))))
    with ThreadPoolExecutor(max_workers=5) as pool:
        return [result for result in pool.map(probe_endpoint, candidates) if result is not None]


def endpoint_config(endpoint, model, context=4096):
    item = next((m for m in endpoint['models'] if m['id'] == model), None)
    if item is None:
        raise ValueError('That model is not currently available at the endpoint. Rescan and select a listed model.')
    reported = item.get('context')
    if isinstance(reported, int) and reported > 0:
        context = min(context, reported)
    if not 1024 <= context <= 32768:
        raise ValueError('Endpoint context budget must be between 1024 and 32768 tokens.')
    return {'source': 'endpoint', 'base_url': endpoint['base_url'], 'model': model,
            'provider': endpoint['provider'], 'context': context}
