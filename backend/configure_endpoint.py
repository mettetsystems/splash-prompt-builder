"""Persist setup's explicit endpoint choice using the installed app interpreter."""
import json
import sys
from backend.storage import Store
from backend.endpoints import normalize_url


def main():
    config = json.load(sys.stdin)
    if config.get('source') != 'endpoint' or not config.get('model'):
        raise ValueError('Expected a selected endpoint model.')
    config['base_url'] = normalize_url(config['base_url'])
    Store().set_config('model', config)


if __name__ == '__main__':
    main()
