package com.datarobot.drum;

import com.datarobot.drum.BasePredictor;
import py4j.GatewayServer;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;

import java.lang.reflect.Constructor;
import java.util.Map;

class EntryPointConfig {
    public String className;
    public String predictorName;
    public int portNumber;

    @Override
    public String toString() {
        return String.format("className: %s; predictorName: %s; portNumber: %d", className, predictorName, portNumber);
    }
}

public class PredictorEntryPoint {
    private BasePredictor predictor;

    public PredictorEntryPoint(String className, String predictorName) throws Exception {
        Class<?> clazz = Class.forName(className);

        Class cls[] = new Class[] { String.class };
        Constructor<?> ctor = clazz.getConstructor(cls);
        predictor = (BasePredictor) ctor.newInstance(predictorName);
    }

    public BasePredictor getPredictor() {
        return predictor;
    }

    private static EntryPointConfig parseArgs(String[] args) throws Exception {
        ArgumentParser parser = ArgumentParsers.newFor("ComponentEntryPoint").build()
                                               .defaultHelp(true)
                                               .description("Calculate checksum of given files.");
        parser.addArgument("--class-name")
              .type(String.class)
              .help("Class name to run py4j gateway on");
        parser.addArgument("--port")
              .type(Integer.class)
              .help("Port number to use for py4j gateway");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        Map<String, Object> optionsMap = options.getAttrs();
        EntryPointConfig config = new EntryPointConfig();
        config.className = (String) optionsMap.getOrDefault("class_name", "");
        if (config.className.isEmpty()) {
            throw new Exception("class-name argument must be provided");
        }
        config.portNumber = (int) optionsMap.getOrDefault("port", 0);
        if (config.portNumber == 0) {
            throw new Exception("port must be provided and must be > 0");
        }

        return config;
    }

    public static void main(String[] args) throws Exception {
        int portNumber = -1;

        Logger logger = LogManager.getLogger(PredictorEntryPoint.class);

        try {
            EntryPointConfig config = parseArgs(args);
            portNumber = config.portNumber;
            PredictorEntryPoint entryPoint = new PredictorEntryPoint(config.className, config.predictorName);
            logger.info("Starting py4j GatewayServer using: class name: {}; port: {}", config.className, config.portNumber);
            // TODO: use a specific port, note multiple such gateways might be running.
            GatewayServer gatewayServer = new GatewayServer(entryPoint, config.portNumber);
            gatewayServer.start();
        } catch (py4j.Py4JNetworkException e) {
            logger.error(String.format("PredictorEntryPoint failed to start py4j GatewayServer on port: %d; Message: %s", portNumber, e.getMessage()), e);
            System.exit(1);
        }
    }
}
