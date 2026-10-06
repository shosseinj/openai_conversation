"""Portable port cleanup, development HTTPS and optional browser opening."""
from datetime import datetime, timedelta, timezone
import ipaddress
import os
from pathlib import Path
import socket
import time


def free_port(port):
    """Stop TCP listeners on this port; never terminate client-only connections."""
    import psutil

    def listeners():
        connections = psutil.net_connections(kind='tcp')
        return [c for c in connections if c.status == psutil.CONN_LISTEN and c.laddr.port == port]

    try:
        occupied = listeners()
        if not occupied:
            return
        if any(c.pid is None or c.pid == os.getpid() for c in occupied):
            raise RuntimeError(f'Cannot identify or stop the listener on port {port}')
        processes = [psutil.Process(pid) for pid in {c.pid for c in occupied}]
        print(f'Port {port} is busy; stopping listener PID(s): '+', '.join(str(p.pid) for p in processes), flush=True)
        for process in processes:
            try: process.terminate()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for process in alive:
            print(f'PID {process.pid} did not stop; forcing termination.', flush=True)
            try: process.kill()
            except psutil.NoSuchProcess: pass
        psutil.wait_procs(alive, timeout=2)
        if listeners():
            raise RuntimeError(f'Port {port} is still busy; check process ownership')
    except psutil.AccessDenied as exc:
        raise RuntimeError(f'Cannot free port {port}: permission denied for its listener') from exc
    except psutil.NoSuchProcess:
        if listeners():
            raise RuntimeError(f'Port {port} changed owners; try starting again')


def ensure_https(root, names=None):
    """Keep existing certificates; create a local CA and server pair on first run."""
    folder = Path(root) / 'certs'
    keyfile, certfile = folder / 'server.key', folder / 'server.crt'
    if keyfile.is_file() and certfile.is_file():
        return keyfile, certfile
    if keyfile.exists() or certfile.exists():
        raise RuntimeError('Incomplete HTTPS certificate pair in certs/: restore server.key and server.crt')
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    import psutil

    if names is None:
        names = {'localhost', '127.0.0.1', '::1', socket.gethostname()}
        names.update(n.strip() for n in os.environ.get('HTTPS_NAMES', '').split(',') if n.strip())
        for addresses in psutil.net_if_addrs().values():
            names.update(a.address.split('%')[0] for a in addresses if a.family in (socket.AF_INET, socket.AF_INET6))
    alternatives = []
    for name in sorted(set(names)):
        try: alternatives.append(x509.IPAddress(ipaddress.ip_address(name)))
        except ValueError: alternatives.append(x509.DNSName(name))
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Persian Voice local development CA')])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    folder.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'wb') as output:
        output.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    certfile.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (folder / 'ca.crt').write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    print('Created development HTTPS certificates in certs/. Trust certs/ca.crt in your browser to use the microphone.', flush=True)
    return keyfile, certfile


def open_browser_when_ready(url, root):
    import ssl
    import urllib.error
    import urllib.request
    import webbrowser
    cafile = Path(root) / 'certs' / 'ca.crt'
    context = ssl.create_default_context(cafile=str(cafile) if cafile.is_file() else None)
    for _ in range(240):
        try:
            with urllib.request.urlopen(url + '/health', context=context, timeout=1):
                webbrowser.open(url)
                return
        except (OSError, urllib.error.URLError):
            time.sleep(.5)
    print(f'Open the frontend manually: {url}', flush=True)
