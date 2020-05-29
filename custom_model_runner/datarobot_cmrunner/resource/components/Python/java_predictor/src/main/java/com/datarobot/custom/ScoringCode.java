package com.datarobot.custom;

import com.datarobot.prediction.Predictors;
import com.datarobot.prediction.IClassificationPredictor;
import com.datarobot.prediction.IPredictorInfo;
import com.datarobot.prediction.IRegressionPredictor;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import java.io.File;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.StringWriter;
import java.io.StringReader;
import java.io.PrintWriter;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

import java.io.FileNotFoundException;
import java.io.IOException;

import com.datarobot.custom.BasePredictor;

public class ScoringCode extends BasePredictor
{
    private IPredictorInfo model;
    private boolean isRegression;
    private String negativeClassLabel = null;
    private String positiveClassLabel = null;
    private Map<String, Object> params = null;

    public ScoringCode(String name) {
        super(name);
    }

    private static IPredictorInfo loadModel(String modelDir) throws IOException {
        var urls = new ArrayList<URL>();
        for (var file : Paths.get(modelDir).toFile().listFiles()) {
            if (file.isFile() && file.getName().endsWith(".jar")) {
                urls.add(new URL("file://" + file.getCanonicalPath()));
            }
        }
        var urlClassLoader = new URLClassLoader(urls.toArray(new URL[urls.size()]));
        return Predictors.getPredictor(urlClassLoader);
    }

    public String predict(String stringCSV) throws Exception {
        List<?> predictions = null;
        CSVPrinter csvPrinter = null;
        try {
            predictions = this.scoreStringCSV(stringCSV);

            if (this.isRegression) {
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader("Predictions"));
                for (var value : predictions) {
                    csvPrinter.printRecord(value);
                }
            } else {
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(this.positiveClassLabel, this.negativeClassLabel));
                for (var val : predictions) {
                    HashMap<String, Double> value = (HashMap<String, Double>) val;
                    csvPrinter.printRecord(value.get(this.positiveClassLabel), value.get(this.negativeClassLabel));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        csvPrinter.flush();
        StringWriter outStream = (StringWriter) csvPrinter.getOut();
        return outStream.toString();
    }

    public void configure(Map<String, Object> params) {
        this.params = params;

        String customModelPath = (String) this.params.get("__custom_model_path__");
        this.negativeClassLabel = (String) this.params.get("negativeClassLabel");
        this.positiveClassLabel = (String) this.params.get("positiveClassLabel");

        try {
            this.model = loadModel(customModelPath);
            this.isRegression = this.model instanceof IRegressionPredictor;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<?> scoreStringCSV(String stringCSV) throws IOException {
        var predictions = new ArrayList<>();
        var csvFormat = CSVFormat.DEFAULT.withHeader();

        try (var parser = csvFormat.parse(new BufferedReader(new StringReader(stringCSV)))) {
            for (var csvRow : parser) {
                var mapRow = csvRow.toMap();
                predictions.add(scoreRow(mapRow));
            }
        }
        return predictions;
    }

    private Object scoreRow(Map<String, ?> row) {
        if (isRegression) {
            return ((IRegressionPredictor) model).score(row);
        } else {
            var prediction = ((IClassificationPredictor) model).score(row);
            var originalClassLabels = ((IClassificationPredictor) model).getClassLabels();
            if (originalClassLabels.length == 2) {
                var remappedPrediction = new HashMap<String, Double>();
                remappedPrediction.put(this.negativeClassLabel, prediction.get(originalClassLabels[1]));
                remappedPrediction.put(this.positiveClassLabel, prediction.get(originalClassLabels[0]));
                prediction = remappedPrediction;
            }
            return prediction;
        }
    }
}
