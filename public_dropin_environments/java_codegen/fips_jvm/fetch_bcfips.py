"""Download the Bouncy Castle FIPS jars (RAPTOR-21063) from Maven Central into the given directory,
checking each against a pinned SHA-256. They are the same versions and the same bytes as Chainguard's
jdk-fips image ships (bouncycastle-fips 2.0.0, FIPS 140-3), so no extra base image is needed.

usage: python fetch_bcfips.py <dest dir>
Set MAVEN_REPO_URL to download from a Maven mirror instead of Maven Central.
"""

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

MAVEN = os.environ.get("MAVEN_REPO_URL") or "https://repo1.maven.org/maven2"
ATTEMPTS = 3

# (artifact, version, sha256). bc-fips is the validated module; the others are the JSSE (TLS)
# provider and the utilities it needs (bctls-fips' module requires only fips.core and fips.util).
JARS = (
    ("bc-fips", "2.0.0", "f6a25fd5744e91cc019a9f79733ae757da6648a80946fad9932041ad05695240"),
    ("bctls-fips", "2.0.19", "0bceb027307098323b0e1365b31bca19caf4f0f20e787d1b48b40be3e6804be0"),
    ("bcutil-fips", "2.0.3", "d7ab2fa5ba33594324a2cc26c75fd30e6a13e4d6524cb2027bb8384fd1befa14"),
)


def download(url: str) -> bytes:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except OSError as e:
            if attempt == ATTEMPTS:
                raise
            print(f"{url}: {e}, retrying", file=sys.stderr)
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for artifact, version, sha256 in JARS:
        url = f"{MAVEN.rstrip('/')}/org/bouncycastle/{artifact}/{version}/{artifact}-{version}.jar"
        data = download(url)
        digest = hashlib.sha256(data).hexdigest()
        if digest != sha256:
            sys.exit(f"{url}: sha256 {digest}, expected {sha256}")
        (dest / f"{artifact}.jar").write_bytes(data)
        print(f"{artifact}-{version}: sha256 ok")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
