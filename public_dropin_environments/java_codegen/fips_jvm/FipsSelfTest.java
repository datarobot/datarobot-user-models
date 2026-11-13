import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.Provider;
import java.security.Security;
import java.util.Arrays;
import java.util.Set;

import javax.net.ssl.SSLContext;

import org.bouncycastle.crypto.CryptoServicesRegistrar;
import org.bouncycastle.crypto.fips.FipsStatus;

/**
 * JVM FIPS startup self-test (RAPTOR-21063), run from source by start_server.sh before the server
 * starts: java FipsSelfTest.java (the module path and module come from JAVA_TOOL_OPTIONS).
 *
 * Checks that the image's Bouncy Castle FIPS module is the one in effect, in approved-only mode,
 * with default TLS on its JSSE provider, and exits 78 (EX_CONFIG) with a "[datarobot-fips] FATAL"
 * line otherwise. A no-op unless
 * DATAROBOT_FIPS_ENFORCE is "true" (case and surrounding spaces ignored, as in fips/sitecustomize.py).
 */
public final class FipsSelfTest {
    static final int EXIT_CODE = 78;
    static final String MODULE_DIR = "/usr/share/java/bouncycastle-fips/";
    static final Set<String> NON_FIPS_PROVIDERS =
        Set.of("SunJCE", "SunRsaSign", "SunEC", "SunJSSE", "SunPKCS11");

    public static void main(String[] args) {
        String enforce = System.getenv("DATAROBOT_FIPS_ENFORCE");
        if (enforce == null || !"true".equals(enforce.trim().toLowerCase())) {
            return;
        }
        try {
            System.out.println("[datarobot-fips] JVM FIPS self-test passed: " + run());
        } catch (Throwable t) {
            Throwable cause = t.getCause();
            System.err.println("[datarobot-fips] FATAL: JVM FIPS self-test failed, refusing to start: " + t
                + (cause == null ? "" : " <- " + cause));
            System.err.flush();
            Runtime.getRuntime().halt(EXIT_CODE);
        }
    }

    static String run() throws Exception {
        // 1. The module loaded and passed its power-on self-tests (HMAC over the jar plus KATs).
        if (!FipsStatus.isReady()) {
            throw new IllegalStateException("module not ready: " + FipsStatus.getStatusMessage());
        }
        // 2. The module comes from the image's module path (JAVA_TOOL_OPTIONS), so the image layout is
        // the one this check was built for.
        Path jar = Paths.get(FipsStatus.class.getProtectionDomain().getCodeSource().getLocation().toURI())
            .toAbsolutePath().normalize();
        if (!jar.startsWith(MODULE_DIR)) {
            throw new IllegalStateException("module loaded from " + jar + ", not " + MODULE_DIR);
        }
        // 3. BCFIPS is the first provider, so it serves every algorithm it offers.
        Provider first = Security.getProviders()[0];
        if (!"BCFIPS".equals(first.getName())) {
            throw new IllegalStateException("provider 1 is " + first.getName() + ", not BCFIPS");
        }
        // 4. Approved-only mode.
        if (!CryptoServicesRegistrar.isInApprovedOnlyMode()) {
            throw new IllegalStateException("module is not in approved-only mode");
        }
        // 5. None of the JDK's own crypto providers is loaded (the same list apply_bcfips_security.sh
        // removes), so nothing can serve an algorithm outside the FIPS module.
        for (Provider p : Security.getProviders()) {
            if (NON_FIPS_PROVIDERS.contains(p.getName())) {
                throw new IllegalStateException("non-FIPS provider " + p.getName() + " is loaded");
            }
        }
        // 6. Non-approved algorithm blocked: no provider at all may offer HMAC-MD5.
        Provider[] md5 = Security.getProviders("Mac.HmacMD5");
        if (md5 != null) {
            throw new IllegalStateException("HmacMD5 is offered by " + Arrays.toString(md5));
        }
        // 7. Default TLS works on the FIPS JSSE provider (what HttpsURLConnection uses), so a
        // java.security without a usable default key/trust manager algorithm can't ship.
        Provider tls = SSLContext.getDefault().getProvider();
        if (!"BCJSSE".equals(tls.getName())) {
            throw new IllegalStateException("default TLS provider is " + tls.getName() + ", not BCJSSE");
        }
        return first.getInfo() + " from " + jar;
    }
}
