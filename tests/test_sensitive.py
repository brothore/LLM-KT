import importlib.util
import os
import stat
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('sensitivity_audit', PROJECT / 'tools/audit_sensitive.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SensitivityTests(unittest.TestCase):
    def test_credentials_email_phone_and_known_values(self):
        samples = [(b'sk-or-' + b'v1-' + b'a' * 40, 'provider_api_key'),
                   (b'person' + b'@' + b'gmail.com', 'email_requires_review'),
                   (b'138' + b'12345678', 'phone_number'),
                   (b'synthetic-local-credential', 'known_local_secret')]
        for value, category in samples:
            findings = AUDIT.scan(value, 'synthetic', [b'synthetic-local-credential'])
            self.assertIn(category, [f[2] for f in findings])
        safe = b'public@users.noreply.github.com contact@example.com'
        self.assertEqual(AUDIT.scan(safe, 'synthetic', []), [])

    def test_private_paths(self):
        for path in ('data/raw/x.csv', 'data/snapshots/x/test.csv', '.secrets/x.key', '.env.local', 'runs/x/requests/y.json'):
            self.assertTrue(AUDIT.private_path(path))
        for path in ('tests/fixtures/test_quelevel.csv', '.env.example', 'configs/jev.yaml'):
            self.assertFalse(AUDIT.private_path(path))

    def test_deleted_sensitive_history_is_still_detected(self):
        root = PROJECT / 'tests' / ('work_audit_' + uuid.uuid4().hex)
        root.mkdir()
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
        try:
            git('init', '-q', '-b', 'main')
            git('config', 'user.name', 'synthetic-test')
            git('config', 'user.email', 'synthetic@users.noreply.github.com')
            path = root / 'sample.txt'
            path.write_bytes(b'sk-or-' + b'v1-' + b'a' * 40)
            git('add', 'sample.txt')
            git('commit', '-q', '-m', 'Synthetic scanner fixture')
            path.write_text('removed synthetic value')
            git('add', 'sample.txt')
            git('commit', '-q', '-m', 'Remove fixture value')
            with patch.object(AUDIT, 'ROOT', root), patch.object(AUDIT, 'local_secret_values', return_value=[]):
                _, findings = AUDIT.audit(worktree=True, staged=True, history=True)
            self.assertTrue(any(f[0].startswith('git-object:') and f[2] == 'provider_api_key' for f in findings))
        finally:
            resolved = root.resolve()
            assert resolved.parent == (PROJECT / 'tests').resolve() and resolved.name.startswith('work_audit_')
            def clear_readonly(function, path, exc_info):
                candidate = Path(path).resolve()
                assert candidate.is_relative_to(resolved)
                os.chmod(candidate, stat.S_IWRITE | stat.S_IREAD)
                function(path)
            shutil.rmtree(resolved, onerror=clear_readonly)


if __name__ == '__main__':
    unittest.main()
