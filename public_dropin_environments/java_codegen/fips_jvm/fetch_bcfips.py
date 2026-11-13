"""Download the Bouncy Castle FIPS jars (RAPTOR-21063) from Maven Central into the given directory,
checking each against a pinned SHA-256. Same versions as DataRobot's other FIPS Java images (Chainguard
jre-fips): bc-fips 2.1.3, the patched release of the 2.1 line (2.1.1 has interim FIPS 140-3
certificate #4943), which fixes the CVEs in every validated version. Revisit when the pending BC-FJA
CMVP submissions land.

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

# (artifact, version, sha256). bc-fips is the FIPS module; the others are the JSSE (TLS) provider,
# the utilities it needs, and PKIX, kept to match the other FIPS Java images.
JARS = (
    ("bc-fips", "2.1.3", "f55fc476ede820d5c03fe87e6ced5bb0095e1dc30b387445846a3ff0274f8c0e"),
    ("bctls-fips", "2.1.25", "af604aa5b337725872651067a819ebc03aa298685572b0b2ad5343c786faad0a"),
    ("bcutil-fips", "2.1.8", "f44904f61552c823bcc3686a24052ee3135212d547754eaa2ff6532e210103cb"),
    ("bcpkix-fips", "2.1.13", "cf161ba2f6b4505816f4661663b3e0771184a50ee5b178a0018c817b336f0e5c"),
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
