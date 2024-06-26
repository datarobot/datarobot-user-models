package com.mlpiper.components.test_java_standalone;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;


public class JavaStandaloneComponent
{
    public static void main( String[] args ) throws Exception {
        System.out.println("JavaStandaloneComponent");
        ArgumentParser parser = ArgumentParsers.newFor("Checksum").build()
                                               .defaultHelp(true)
                                               .description("Calculate checksum of given files.");
        parser.addArgument("--exit-value")
              .type(Integer.class)
              .setDefault(0)
              .help("Exit status of component");
        parser.addArgument("--iter")
              .type(Integer.class)
              .setDefault(5)
              .help("Number of 1 second iterations to perform");

        parser.addArgument("--input-model")
              .help("Path to input model to consume");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        System.out.println(options);

        int exitValue = options.get("exit_value");
        int iter = options.get("iter");
        String modelPathStr = options.get("input_model");

        System.out.println("exitValue:    " + exitValue);
        System.out.println("Iter:         " + iter);
        System.out.println("modelPath:    " + modelPathStr);

        Paths.get(modelPathStr);
        Path modelFile = Paths.get(modelPathStr);
        if (!modelFile.toFile().exists()) {
            throw new Exception("Model file does exits");
        }

        System.out.println();
        System.out.println("Loop:");
        for (int i=0 ; i < iter ; i++) {
            System.out.println(String.format("Iter: %d/%d", i, iter));
            TimeUnit.SECONDS.sleep(1);
        }

        System.out.println("About to exit with: " + exitValue);
        if (exitValue != 0) {
            System.exit(exitValue);
        }
    }
}
