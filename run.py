"""Automatic cross-platform launcher. Run python run.py or use run.bat/run.sh."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Set up and launch the Persian OpenAI voice app')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8500)
    parser.add_argument('--enable-local', action='store_true', help='Explicitly enable the existing Local installation')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--skip-install', action='store_true', help='Skip package installation after initial setup')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if sys.version_info < (3, 10):
        print('Python 3.10 or newer is required; Python 3.12 is recommended.', file=sys.stderr)
        return 1
    folder = ROOT / '.venv'
    python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    try:
        if not folder.exists():
            print('Creating .venv for this fresh checkout...', flush=True)
            venv.EnvBuilder(with_pip=True).create(folder)
        if not python.is_file():
            raise RuntimeError('Existing .venv has no Python executable for this OS. Restore it or use a separate checkout; it will not be overwritten.')
        print(f'Using existing environment: {python}', flush=True)
        if not args.skip_install:
            print('Installing/checking OpenAI-only application dependencies...', flush=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirement.txt')], cwd=ROOT, check=True)
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(folder)
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        env['ENABLE_LOCAL_STT'] = '1' if args.enable_local else '0'
        command = [str(python), '-B', str(ROOT / 'app.py'), '--host', args.host, '--port', str(args.port)]
        if args.enable_local: command.append('--enable-local')
        if not args.no_browser: command.append('--open-browser')
        print('Starting '+('OpenAI with optional Local models.' if args.enable_local else 'OpenAI only; Local models stay unloaded.'), flush=True)
        print(f'Any existing TCP listener on port {args.port} will be stopped. Press Ctrl+C to stop the app.', flush=True)
        return subprocess.run(command, cwd=ROOT, env=env).returncode
    except KeyboardInterrupt:
        print('\nStopped.', flush=True)
        return 130
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f'Startup failed: {exc}', file=sys.stderr)
        print('See README.md troubleshooting. The existing environment has not been replaced.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
