"""Immutable stage manifests and per-invocation audit records for local artifacts."""
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .common import digest, file_hash, read_json, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def unique_id(prefix):
    return prefix + '_' + uuid.uuid4().hex


def code_identity():
    package = Path(__file__).parent
    files = {p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob('*.py'))}
    result = {'source_sha256': digest(files), 'python': platform.python_version(), 'platform': platform.system()}
    try:
        completed = subprocess.run(['git', '-C', str(package), 'rev-parse', 'HEAD'], capture_output=True, timeout=5)
        if completed.returncode == 0:
            result['git_commit'] = completed.stdout.decode().strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def finish_stage(root, stage, identity, files, parents=()):
    root = Path(root)
    hashes = {name: file_hash(root / name) for name in files}
    payload = {'schema_version': 1, 'stage': stage, 'identity': identity, 'parents': list(parents), 'files': hashes}
    manifest = {**payload, 'artifact_id': digest(payload), 'created_at': utc_now(), 'code': code_identity()}
    write_json(root / 'artifact_manifest.json', manifest)
    return manifest


def verify_stage(root):
    root = Path(root)
    manifest = read_json(root / 'artifact_manifest.json')
    payload = {k: manifest[k] for k in ('schema_version', 'stage', 'identity', 'parents', 'files')}
    if digest(payload) != manifest['artifact_id']:
        raise ValueError('Artifact manifest identity mismatch')
    for name, checksum in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != checksum:
            raise ValueError(f'Artifact missing or corrupted: {name}')
    return manifest


def record_event(root, stage, artifact_id, cache_hit, details=None):
    event = {'invocation_id': unique_id('inv'), 'stage': stage, 'artifact_id': artifact_id,
             'cache_hit': cache_hit, 'at': utc_now(), 'details': details or {}, 'code': code_identity()}
    write_json(Path(root) / 'events' / (event['invocation_id'] + '.json'), event)
    return event
