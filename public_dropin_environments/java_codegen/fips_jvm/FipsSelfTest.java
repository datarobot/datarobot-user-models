import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.Provider;
import java.security.Security;
import java.util.Arrays;

import org.bouncycastle.crypto.CryptoServicesRegistrar;
import org.bouncycastle.crypto.fips.FipsStatus;

/**
 * JVM FIPS startup self-test (RAPTOR-21063), run from source by start_server.sh before the server
 * starts: java --add-modules org.bouncycastle.fips.core FipsSelfTest.java
 *
 * Checks that the image's Bouncy Castle FIPS module is the one in effect, in approved-only mode,
 * and exits 78 (EX_CONFIG) with a "[datarobot-fips] FATAL" line otherwise. A no-op unless
 * DATAROBOT_FIPS_ENFORCE is "true" (case and surrounding spaces ignored, as in fips/sitecustomize.py).
 */
public final class FipsSelfTest {
    static final int EXIT_CODE = 78;
    static final String MODULE_DIR = "/usr/share/java/bouncycastle-fips/";

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
        // 2. The module is the image's copy, not one bundled by the model.
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
        // 5. Non-approved algorithm blocked: no provider at all may offer HMAC-MD5.
        Provider[] md5 = Security.getProviders("Mac.HmacMD5");
        if (md5 != null) {
            throw new IllegalStateException("HmacMD5 is offered by " + Arrays.toString(md5));
        }
        return first.getInfo() + " from " + jar;
    }
}
