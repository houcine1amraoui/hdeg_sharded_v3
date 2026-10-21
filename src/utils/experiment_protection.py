"""Small write guards for HDEG experiments; no model or configuration changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

MARKER = 'HDEG_OUTPUTS_PROTECTED.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def assert_output_unprotected(processed_dir, runner):
    marker = Path(processed_dir)/MARKER
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(
            f'{runner}: existing experiment outputs are protected by {marker}. '
            'No artifacts were written. Use a separately configured experiment; '
            'do not remove this marker to reuse the old outputs.')


def require_new_file(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')


def write_json_new(path, payload):
    path = Path(path)
    require_new_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fresh_run_directory(parent, explicit=None):
    """Reserve a new directory; even an existing empty directory is rejected."""
    path = Path(explicit) if explicit is not None else Path(parent)/(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12])
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def copy_checkpoint_new(source, destination, expected_hash):
    source, destination = Path(source), Path(destination)
    if sha256_file(source) != expected_hash:
        raise ValueError('Source checkpoint differs from expected SHA256')
    require_new_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation is supported on Drive mounts, unlike hard-link publication.
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024*1024)
    if sha256_file(destination) != expected_hash or sha256_file(source) != expected_hash:
        raise RuntimeError('Checkpoint changed during copy or copy verification failed; run is incomplete')
