#!/usr/bin/env python3
"""Upload and verify a complete draft before making a stable release public."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def run(*args):
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish(repo, tag, commit, assets):
    require(re.fullmatch(r'[\w.-]+/[\w.-]+', repo), 'invalid repository')
    require(re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+', tag), 'invalid stable tag')
    require(len({p.name for p in assets}) == len(assets), 'duplicate asset names')
    expected = {p.name: sha(p) for p in assets}
    route = f'repos/{repo}'

    def check_tag():
        require(json.loads(run('gh', 'api', f'{route}/commits/{tag}'))['sha'] == commit,
                'remote release tag moved')

    def verify_downloads():
        release = json.loads(run('gh', 'api', f'{route}/releases/tags/{tag}'))
        require(not release['prerelease'], 'unexpected prerelease')
        names = [asset['name'] for asset in release['assets']]
        require(len(names) == len(expected) and set(names) == set(expected), 'release asset set differs')
        with tempfile.TemporaryDirectory(prefix='ironmlx-release-download-') as tmp:
            run('gh', 'release', 'download', tag, '--repo', repo, '--dir', tmp)
            require({p.name: sha(p) for p in Path(tmp).iterdir()} == expected,
                    'downloaded release assets differ')
        return release

    check_tag()
    # gh create rejects an existing tag release; never delete/overwrite on retry.
    run('gh', 'release', 'create', tag, '--repo', repo, '--verify-tag', '--draft',
        '--title', f'IronMLX {tag}', '--generate-notes')
    run('gh', 'release', 'upload', tag, '--repo', repo, *assets)
    release = verify_downloads()
    require(release['draft'], 'release became public before verification')
    check_tag()
    run('gh', 'api', f"{route}/releases/{release['id']}", '--method', 'PATCH',
        '-F', 'draft=false', '-f', 'make_latest=true')
    require(not verify_downloads()['draft'], 'release is still a draft')
    print(f'Published and downloaded verified stable release: {tag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--repository', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    run(root / 'scripts/release-legal-gate.sh')
    run('python3', root / 'scripts/verify-release-identity.py', args.tag, root / 'dist/IronMLX.app')
    run(root / 'scripts/verify-app-bundle.sh', root / 'dist/IronMLX.app')
    run('xcrun', 'stapler', 'validate', root / 'dist/IronMLX.app')
    run('spctl', '--assess', '--type', 'execute', root / 'dist/IronMLX.app')
    run('python3', root / 'scripts/release-archives.py', 'verify', root / 'dist/IronMLX.app',
        root / '.build/stable-release')
    for dmg in (root / '.build/stable-release').glob('*.dmg'):
        run('codesign', '--verify', '--strict', dmg)
        run('xcrun', 'stapler', 'validate', dmg)
        run('spctl', '--assess', '--type', 'open', '--context', 'context:primary-signature', dmg)
    assets = sorted(p for p in (root / '.build/stable-release').iterdir() if p.is_file())
    update = root / '.build/app-update'
    data = json.loads((update / 'update.json').read_text())
    require(data['tag'] == args.tag and data['channel'] == 'stable', 'update identity mismatch')
    require(data['archive'] == f'IronMLX-{args.tag}-update.zip' and data['feed'] == 'stable.xml',
            'invalid update asset names')
    for kind in ('archive', 'feed'):
        require(sha(update / data[kind]) == data[kind + '_sha256'], 'update hash mismatch')
    assets += [update / data['archive'], update / data['feed'], update / 'update.json']
    manifest = root / '.build/RELEASE-SHA256SUMS'
    manifest.write_text(''.join(f'{sha(p)}  {p.name}\n' for p in assets))
    publish(args.repository, args.tag, run('git', '-C', root, 'rev-parse', 'HEAD'), assets + [manifest])


if __name__ == '__main__':
    main()
