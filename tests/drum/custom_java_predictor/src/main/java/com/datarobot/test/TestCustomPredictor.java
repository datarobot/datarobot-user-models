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

public class TestCustomPredictor extends BasePredictor {
    private String negativeClassLabel = null;
    private String positiveClassLabel = null;
    private String[] classLabels = null;
    private Map<String, Object> params = null;
    private boolean isRegression = false;

    private String inputDataset = null;
    private String outputDataset = null;

    public TestCustomPredictor(String name) {
        super(name);
    }

    public void configure(Map<String, Object> params) {
        this.params = params;

        String customModelPath = (String) this.params.get("__custom_model_path__");
        this.isRegression = this.params.get("target_type").equals("regression");

        this.negativeClassLabel = (String) this.params.get("negativeClassLabel");
        this.positiveClassLabel = (String) this.params.get("positiveClassLabel");
        this.classLabels = (String[]) this.params.get("classLabels");
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

    public String predict(byte[] inputBytes) throws IOException {
        return this.predictReader(new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes))));
    }

    public String predictCSV(String inputFilename) throws IOException {
        return this.predictReader(new BufferedReader(new FileReader(new File(inputFilename))));
    }

    public static void main(String[] args) throws IOException {
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
