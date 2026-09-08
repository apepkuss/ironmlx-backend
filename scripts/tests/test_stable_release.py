"""Release failure boundaries. External Apple/GitHub operations are simulated."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stable', SCRIPTS / 'publish-stable-release.py')
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)


class PublicationTests(unittest.TestCase):
    def scenario(self, failure=None):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'asset.zip'
            asset.write_bytes(b'verified artifact')
            calls = []
            public = False

            def run(*args):
                nonlocal public
                calls.append(args)
                if 'commits/v1.0.0' in str(args):
                    moved = failure == 'tag-moved' and sum('commits/v1.0.0' in str(c) for c in calls) > 1
                    return json.dumps({'sha': 'other' if moved else 'commit'})
                if args[1:3] == ('release', 'upload') and failure == 'upload':
                    raise RuntimeError('upload failed')
                if args[1:3] == ('release', 'download'):
                    Path(args[-1], 'asset.zip').write_bytes(b'corrupt' if failure == 'corrupt' else asset.read_bytes())
                    return ''
                if '--method' in args:
                    public = True
                    return '{}'
                if '/releases/tags/' in str(args):
                    return json.dumps(dict(id=10, draft=not public, prerelease=False,
                                           assets=[{'name': 'extra' if failure == 'asset-set' else 'asset.zip'}]))
                return ''

            with patch.object(stable, 'run', run):
                if failure:
                    with self.assertRaises((ValueError, RuntimeError)):
                        stable.publish('owner/repo', 'v1.0.0', 'commit', [asset])
                    self.assertFalse(public)
                else:
                    stable.publish('owner/repo', 'v1.0.0', 'commit', [asset])
                    self.assertTrue(public)
                    self.assertEqual(sum(c[1:3] == ('release', 'download') for c in calls), 2)

    def test_draft_verified_before_promotion_and_public_download(self):
        self.scenario()

    def test_failure_never_promotes_draft(self):
        for failure in ('upload', 'corrupt', 'asset-set', 'tag-moved'):
            with self.subTest(failure=failure):
                self.scenario(failure)


class SigningTests(unittest.TestCase):
    def scenario(self, status, kind="app"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / f'IronMLX.{kind}'
            if kind == 'app':
                app.mkdir()
            else:
                app.write_bytes(b'dmg')
            bindir = root / 'bin'
            bindir.mkdir()
            log = root / 'calls.jsonl'
            shim = bindir / 'shim'
            shim.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps([name, *args])+'\\n')
if name == 'openssl': print('temporary-test-password')
if name == 'codesign' and '-dv' in args: print('TeamIdentifier=TESTTEAM')
if name == 'xcrun' and args[:2] == ['notarytool', 'submit']:
    print(json.dumps({'status': os.environ['NOTARY_STATUS']}))
''')
            shim.chmod(0o755)
            for name in ('security', 'openssl', 'xcrun', 'codesign', 'plutil', 'ditto', 'spctl'):
                (bindir / name).symlink_to(shim)
            env = os.environ.copy()
            env.update(PATH=f'{bindir}:{env["PATH"]}', RUNNER_TEMP=tmp, CALL_LOG=str(log), NOTARY_STATUS=status,
                       IRONMLX_DEVELOPER_ID_P12_BASE64='dGVzdA==', IRONMLX_DEVELOPER_ID_P12_PASSWORD='test',
                       IRONMLX_SIGNING_IDENTITY='Developer ID Application: Test (TESTTEAM)',
                       IRONMLX_APPLE_TEAM_ID='TESTTEAM', IRONMLX_NOTARY_KEY_ID='test',
                       IRONMLX_NOTARY_ISSUER_ID='test', IRONMLX_NOTARY_PRIVATE_KEY='test')
            # Copy script/resources so simulated receipts stay outside the repository.
            scripts = root / 'scripts'
            scripts.mkdir()
            script = scripts / 'sign-notarize-app.sh'
            script.write_bytes((SCRIPTS / script.name).read_bytes())
            result = subprocess.run(['bash', str(script), str(app)], env=env, capture_output=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            staple = [c for c in calls if c[:3] == ['xcrun', 'stapler', 'staple']]
            self.assertEqual(result.returncode == 0, status == 'Accepted', result.stderr)
            self.assertEqual(bool(staple), status == 'Accepted')
            self.assertFalse((root / 'ironmlx-signing').exists())
            self.assertTrue(any(c[:2] == ['security', 'delete-keychain'] for c in calls))
            signed = [i for i, c in enumerate(calls) if c[:2] == ['codesign', '--force']]
            self.assertEqual(len(signed), 8 if kind == "app" else 1)
            self.assertTrue(all(i < signed[0] for i, c in enumerate(calls) if c[0] == 'plutil'))
            if kind == 'app':
                self.assertIn('--entitlements', calls[signed[1]])
            submit = next(i for i, c in enumerate(calls) if c[:3] == ['xcrun', 'notarytool', 'submit'])
            self.assertGreater(submit, signed[-1])

    def test_missing_credentials_fail_before_keychain_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / 'IronMLX.app'
            app.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith('IRONMLX_')}
            env['RUNNER_TEMP'] = tmp
            result = subprocess.run(['bash', str(SCRIPTS / 'sign-notarize-app.sh'), str(app)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing IRONMLX_DEVELOPER_ID_P12_BASE64', result.stderr)
            self.assertFalse((Path(tmp) / 'ironmlx-signing').exists())

    def test_signing_order_and_accepted_ticket(self):
        self.scenario('Accepted')

    def test_dmg_notarization_and_stapling(self):
        self.scenario('Accepted', 'dmg')

    def test_notary_rejection_cleans_credentials_without_stapling(self):
        self.scenario('Invalid')


if __name__ == '__main__':
    unittest.main()
