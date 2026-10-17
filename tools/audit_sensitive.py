"""Pre-publication audit. Reports locations/categories, never matched values.

Pattern checks are not a substitute for human review of private information.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'provider_api_key': re.compile(rb'(?:sk-or-v1-[A-Za-z0-9_-]{20,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,})'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,})'),
    'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'credential_assignment': re.compile(rb'''(?i)["']?(?:api_key|access_token|client_secret|password|cdkey|cd_key)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=-]{24,}["']'''),
    'personal_user_directory': re.compile(rb'(?i)(?:[A-Z]:[/\\]Users[/\\](?!Public\b|Default\b|<)[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+)'),
    'phone_number': re.compile(rb'(?<!\d)1[3-9]\d{9}(?!\d)'),
    'product_cd_key': re.compile(rb'\b(?:[A-Z0-9]{5}-){4}[A-Z0-9]{5}\b'),
}
EMAIL = re.compile(rb'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
ALLOWED_EMAIL_DOMAINS = {b'example.com', b'example.org', b'example.invalid', b'users.noreply.github.com'}
DENIED_DIRECTORIES = {'.secrets', 'secrets', '.venv', 'runs', 'build', 'dist', '__pycache__'}
PRIVATE_DATA = {'raw', 'processed', 'snapshots', 'prepared', 'reports'}


def git(*args):
    process = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True)
    if process.returncode:
        raise RuntimeError('Git command failed; run from an initialized repository')
    return process.stdout


def private_path(path):
    parts = PurePosixPath(path).parts
    return (any(p in DENIED_DIRECTORIES for p in parts) or
            (len(parts) > 1 and parts[0] == 'data' and parts[1] in PRIVATE_DATA) or
            (PurePosixPath(path).name.startswith('.env') and PurePosixPath(path).name != '.env.example') or
            PurePosixPath(path).suffix.lower() in {'.zip', '.key', '.pem', '.p12', '.pfx', '.sqlite', '.log'})


def local_secret_values():
    # Read only into memory to catch known local keys even if their format is unknown.
    values = [os.environ.get(name, '').encode() for name in ('OPENROUTER_API_KEY', 'OPENAI_API_KEY', 'TYPESAFE_API_KEY')]
    path = ROOT.parent / '.secrets/openrouter.key'
    if path.is_file():
        values.append(path.read_text(encoding='utf-8-sig').strip().encode())
    return [v for v in values if len(v) >= 12]


def scan(content, label, secrets):
    findings = []
    for category, pattern in PATTERNS.items():
        match = pattern.search(content)
        if match:
            findings.append((label, content[:match.start()].count(b'\n') + 1, category))
    for match in EMAIL.finditer(content):
        if match.group().rsplit(b'@', 1)[1].lower() not in ALLOWED_EMAIL_DOMAINS:
            findings.append((label, content[:match.start()].count(b'\n') + 1, 'email_requires_review'))
    for value in secrets:
        if value in content:
            findings.append((label, content[:content.index(value)].count(b'\n') + 1, 'known_local_secret'))
    return findings


def audit(worktree=False, staged=False, history=False):
    secrets = local_secret_values()
    findings, counts = [], {'worktree_files': 0, 'staged_files': 0, 'git_objects': 0}
    for identity in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
        ident = git('var', identity)
        match = EMAIL.search(ident)
        if not match or not match.group().lower().endswith(b'@users.noreply.github.com'):
            findings.append(('git-config:' + identity, 0, 'non_noreply_commit_identity'))
    if worktree:
        paths = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').split(b'\0')
        for raw in sorted(set(p for p in paths if p)):
            name = raw.decode('utf-8')
            path = ROOT / name
            if path.is_file():
                counts['worktree_files'] += 1
                if private_path(name):
                    findings.append(('worktree:' + name, 0, 'private_file_path'))
                findings.extend(scan(path.read_bytes(), 'worktree:' + name, secrets))
    if staged:
        entries = git('ls-files', '--stage', '-z').split(b'\0')
        for entry in entries:
            if not entry:
                continue
            metadata, raw_name = entry.split(b'\t', 1)
            mode, oid, stage = metadata.split()
            name = raw_name.decode('utf-8')
            counts['staged_files'] += 1
            if stage != b'0' or mode in {b'160000', b'120000'}:
                findings.append(('index:' + name, 0, 'unmerged_or_external_file'))
                continue
            if private_path(name):
                findings.append(('index:' + name, 0, 'private_file_path'))
            findings.extend(scan(git('cat-file', 'blob', oid.decode()), 'index:' + name, secrets))
    if history:
        # Includes dangling objects, unlike scanning only reachable commits.
        objects = git('cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)').splitlines()
        for record in objects:
            oid, kind = record.split()
            counts['git_objects'] += 1
            if kind in {b'blob', b'commit', b'tag'}:
                findings.extend(scan(git('cat-file', kind.decode(), oid.decode()), 'git-object:' + oid.decode(), secrets))
            elif kind == b'tree':
                for raw_name in git('ls-tree', '-r', '--name-only', '-z', oid.decode()).split(b'\0'):
                    if raw_name and private_path(raw_name.decode('utf-8')):
                        findings.append(('git-tree:' + oid.decode(), 0, 'private_file_path'))
        for line in git('log', '--all', '--format=%H%x00%ae%x00%ce').splitlines():
            fields = line.split(b'\0')
            if len(fields) == 3:
                for email in fields[1:]:
                    if email and not email.lower().endswith(b'@users.noreply.github.com'):
                        findings.append(('commit:' + fields[0].decode(), 0, 'non_noreply_commit_identity'))
    return counts, sorted(set(findings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', action='store_true')
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    if not any((args.worktree, args.staged, args.history)):
        parser.error('Choose --worktree, --staged and/or --history')
    try:
        counts, findings = audit(args.worktree, args.staged, args.history)
    except (RuntimeError, OSError) as exc:
        print('Audit could not complete:', type(exc).__name__)
        return 2
    print('Sensitivity audit counts:', counts)
    for label, line, category in findings:
        print(f'BLOCK: {label}:{line} [{category}]')
    if findings:
        print('FAILED. Remove/review flagged content and clean affected Git objects before publishing.')
        return 1
    print('PASS. No detected secrets/personal identifiers. Manual privacy review is still required.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
