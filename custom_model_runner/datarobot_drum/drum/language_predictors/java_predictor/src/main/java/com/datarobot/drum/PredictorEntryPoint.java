package com.datarobot.drum;

import org.apache.log4j.PatternLayout;
import com.datarobot.drum.BasePredictor;
import py4j.GatewayServer;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;
import org.apache.log4j.ConsoleAppender;


import java.lang.reflect.Constructor;
import java.util.Map;

class EntryPointConfig {
    public String className;
    public String predictorName;
    public int portNumber;

    @Override
    public String toString() {
        return String.format("className: %s, predictorName: %s portNumber: %d", className, predictorName, portNumber);
    }
}

public class PredictorEntryPoint {
    private BasePredictor predictor;

    public PredictorEntryPoint(String className, String predictorName) throws Exception {
        ConsoleAppender console = new ConsoleAppender(); //create appender
        //configure the appender
        String PATTERN = "%d [%p|%c|%C{1}] %m%n";
        console.setLayout(new PatternLayout(PATTERN));
        console.setThreshold(Level.DEBUG);
        console.activateOptions();
        //add appender to any Logger (here is root)
        Logger.getRootLogger().addAppender(console);
        Logger.getRootLogger().setLevel(Level.INFO);

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
        try {
            EntryPointConfig config = parseArgs(args);
            portNumber = config.portNumber;
            PredictorEntryPoint entryPoint = new PredictorEntryPoint(config.className, config.predictorName);
            // TODO: use a specific port, note multiple such gateways might be running.
            GatewayServer gatewayServer = new GatewayServer(entryPoint, config.portNumber);
            gatewayServer.start();
        } catch (Exception e) {
            System.out.println(String.format("PredictorEntryPoint failed to start py4j GatewayServer on port: %d", portNumber));
            e.printStackTrace();
            System.exit(1);
        }
    }
}
