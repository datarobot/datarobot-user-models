package com.datarobot.test;

import com.datarobot.drum.BasePredictor;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.io.IOUtils;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.appender.ConsoleAppender;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.Configurator;
import org.apache.logging.log4j.core.config.LoggerConfig;
import org.apache.logging.log4j.core.layout.PatternLayout;


/***
To run this test from CLI:
compile TestCustomPredictor

run the command:

DRUM_JAVA_CUSTOM_CLASS_PATH=tests/functional/custom_java_predictor/* DRUM_JAVA_CUSTOM_PREDICTOR_CLASS=com.datarobot.test.TestCustomPredictor drum score --code-dir tests/drum/custom_java_predictor/ --target-type regression --input tests/testdata/juniors_3_year_stats_regression.csv
***/


public class TestCustomPredictor extends BasePredictor {
    private String negativeClassLabel = null;
    private String positiveClassLabel = null;
    private String[] classLabels = null;
    private Map<String, Object> params = null;
    private boolean isRegression = false;

    private String inputDataset = null;
    private String outputDataset = null;

    private ObjectMapper objectMapper = new ObjectMapper();

    public TestCustomPredictor(String name) {
        super(name);
    }

    private void initCustomLogger(String loggerName) {
        // Example on how to create a logger with customized pattern and appender
        String PATTERN = "%d [%p|%c|%C{1}] %m%n";

        LoggerContext context = (LoggerContext) LogManager.getContext();
        Configuration config = context.getConfiguration();

        PatternLayout.Builder builder = PatternLayout.newBuilder().withPattern(PATTERN);
        PatternLayout layout = builder.build();

        ConsoleAppender.Builder consoleAppenderBuilder = ConsoleAppender.newBuilder()
            .withName("CONSOLE_APPENDER_NAME")
            .withLayout(layout);
        ConsoleAppender consoleApp = consoleAppenderBuilder.build();
        consoleApp.start();

        LoggerConfig loggerConfig = new LoggerConfig("ConfigName", Level.INFO, false);
        loggerConfig.addAppender(consoleApp, null, null);
        config.addLogger(loggerName, loggerConfig);

        context.updateLoggers(config);
    }

    @Override
    public void configure(Map<String, Object> params) {
        this.params = params;

        String customModelPath = (String) this.params.get("__custom_model_path__");
        this.isRegression = this.params.get("target_type").equals("regression");

        this.negativeClassLabel = (String) this.params.get("negativeClassLabel");
        this.positiveClassLabel = (String) this.params.get("positiveClassLabel");
        this.classLabels = (String[]) this.params.get("classLabels");


        String customLoggerName = "MyCustomLogger";
        initCustomLogger(customLoggerName);
        Logger logger = LogManager.getLogger(TestCustomPredictor.class);
        // logging level is defined by DRUM, to modify, provide --logging-level <level>
        logger.info("Default logger - INFO");
        logger.warn("Default logger - WARN");
        logger.error("Default logger - ERROR");

        Logger customLogger = LogManager.getContext().getLogger(customLoggerName);
        customLogger.info("Customized logger - INFO");
        customLogger.warn("Customized logger - WARN");
        customLogger.error("Customized logger - ERROR");
    }

    public String predictReader(Reader in) throws IOException {
        CSVPrinter csvPrinter = null;

        try {
            var csvFormat = CSVFormat.DEFAULT.withHeader();
            var parser = csvFormat.parse(in);

            if (this.isRegression) {
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader("Predictions"));
                int i = 0;
                for (var csvRow : parser) {
                    csvPrinter.printRecord(++i);
                }
            } else {
                this.classLabels = new String[]{this.positiveClassLabel, this.negativeClassLabel};
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(this.classLabels));
                for (var csvRow : parser) {
                    csvPrinter.printRecord(0.7, 0.3);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        csvPrinter.flush();
        StringWriter outStream = (StringWriter) csvPrinter.getOut();
        return outStream.toString();
    }

    @Override
    public String predict(byte[] inputBytes) throws IOException {
        return this.predictReader(new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes))));
    }

    @Override
    public <T> T predictUnstructured(byte[] inputBytes, String mimetype, String charset, Map<String, String> query) {
        System.out.println("incoming mimetype: " + mimetype);
        System.out.println("Incoming Charset: " + charset);
        try {
            System.out.println("Incoming Query: " + objectMapper.writeValueAsString(query));
        } catch (JsonProcessingException e) { 
            e.printStackTrace();
        }
        
        String retMode = query.getOrDefault("ret_mode", "text");
        String s = null;
        try { 
            s = new String(inputBytes, charset);
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }

        System.out.println("Incoming data: " + s );

        Integer count = 10;
        System.out.println(count.intValue());

        switch (retMode) {
            case "binary":
                byte[] retBytes = null;
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                // ObjectOutputStream out = null;
                DataOutputStream dos = null;
                try {
                    dos = new DataOutputStream(bos);                 
                    dos.writeInt( count );
                    dos.flush();
                    retBytes = bos.toByteArray();
                } catch (IOException e) { 
                    e.printStackTrace() ;
                } finally {
                    try {
                        bos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                return (T) retBytes;
            default:
                String retString = null;
                try {
                    retString = objectMapper.writeValueAsString(count);
                } catch (JsonProcessingException e) {
                    e.printStackTrace();
                }
                return (T) retString;  
        }
    }


    public static void main(String[] args) throws IOException, Exception {
        Map<String, Object> params = new HashMap<String, Object>();
        ArgumentParser parser = ArgumentParsers.newFor("TestCustomPredictor").build()
                .defaultHelp(true)
                .description("");
        parser.addArgument("--model-dir")
                .type(String.class)
                .help("Model dir");
        parser.addArgument("--input")
                .type(String.class)
                .help("Input dataset");
        parser.addArgument("--output")
                .type(String.class)
                .help("Path to output csv");
        parser.addArgument("--target-type")
                .type(String.class)
                .help("Target type");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        Map<String, Object> optionsMap = options.getAttrs();

        params.put("__custom_model_path__", (String) optionsMap.getOrDefault("model_dir", ""));
        params.put("target_type", (String) optionsMap.get("target_type"));
        var outFile = (String) optionsMap.getOrDefault("output", null);
        var inputDataset = (String) optionsMap.getOrDefault("input", "");

        OutputStream out;

        var predictor = new TestCustomPredictor("test custom predictor");
        predictor.configure(params);
        var ret = predictor.predictCSV(inputDataset);
        if (outFile != null) {
            out = new FileOutputStream(outFile);
            out.write(ret.getBytes(StandardCharsets.UTF_8));
            out.close();
        } else {
            System.out.println(ret);
        }
    }
}
