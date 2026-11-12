#!/bin/sh
# Switch the JDK's java.security to the Bouncy Castle FIPS provider in approved-only mode
# (RAPTOR-21063). Same provider list as Chainguard's openjdk-bcfips-policy-140-3-j11 package, but
# applied to this JDK's own file, so its newer TLS and certpath defaults are kept.
set -eu
sec="${1:?usage: apply_bcfips_security.sh <JAVA_HOME>/conf/security/java.security}"
tmp="$sec.bcfips"

awk '
  /^security\.provider\.[0-9]+=/ {
    if (!done) {
      print "security.provider.1=org.bouncycastle.jcajce.provider.BouncyCastleFipsProvider C:DEFRND[SHA256];ENABLE{ALL};"
      print "security.provider.2=org.bouncycastle.jsse.provider.BouncyCastleJsseProvider fips:BCFIPS"
      print "security.provider.3=SUN"
      print "security.provider.4=SunJGSS"
      print "security.provider.5=XMLDSig"
      print "security.provider.6=SunPCSC"
      print "security.provider.7=JdkLDAP"
      print "security.provider.8=JdkSASL"
      done = 1
    }
    next
  }
  /^org\.bouncycastle\.fips\.approved_only=/ { next }
  { print }
  END { print "org.bouncycastle.fips.approved_only=true" }
' "$sec" > "$tmp"

# awk succeeds even when nothing matched, so check the result.
grep -q '^security\.provider\.1=org\.bouncycastle\.jcajce\.provider\.BouncyCastleFipsProvider ' "$tmp"
grep -q '^org\.bouncycastle\.fips\.approved_only=true$' "$tmp"
if grep -qE '^security\.provider\.[0-9]+=(SunJCE|SunRsaSign|SunEC|SunJSSE|SunPKCS11)' "$tmp"; then
  echo "apply_bcfips_security.sh: a non-FIPS crypto provider is still listed in $tmp" >&2
  exit 1
fi
cat "$tmp" > "$sec"
rm -f "$tmp"
