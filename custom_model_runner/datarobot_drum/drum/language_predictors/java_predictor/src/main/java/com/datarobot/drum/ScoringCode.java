package com.datarobot.drum;

import com.datarobot.prediction.IClassificationPredictor;
import com.datarobot.prediction.IPredictorInfo;
import com.datarobot.prediction.IRegressionPredictor;
import com.datarobot.prediction.Predictors;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ScoringCode extends BasePredictor {
    private IPredictorInfo model;
    private boolean isRegression;
    private String negativeClassLabel = null;
    private String positiveClassLabel = null;
    private String[] classLabels = null;
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

    public String predict(byte[] inputBytes) throws Exception {
        String ret = null;

        try {
            ret = this.scoreReader(new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes))));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    public String predictCSV(String inputFilename) throws Exception {
        String ret = null;

        try {
            ret = this.scoreReader(new BufferedReader(new FileReader(new File(inputFilename))));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    private String predictionsToString(List<?> predictions) throws Exception {
        CSVPrinter csvPrinter = null;

        try {
            if (this.isRegression) {
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader("Predictions"));
                for (var value : predictions) {
                    csvPrinter.printRecord(value);
                }
            } else {
                csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(this.classLabels));
                for (var val : predictions) {
                    HashMap<String, Double> value = (HashMap<String, Double>) val;
                    var predRow = Arrays.stream(this.classLabels).map(value::get).toArray(Double[]::new);
                    csvPrinter.printRecord(predRow);
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
        this.classLabels = (String[]) this.params.get("classLabels");

        if (this.negativeClassLabel != null && this.positiveClassLabel != null) {
            this.classLabels = new String[]{this.positiveClassLabel, this.negativeClassLabel};
        }

        try {
            this.model = loadModel(customModelPath);
            this.isRegression = this.model instanceof IRegressionPredictor;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String scoreReader(Reader in) throws IOException, Exception {
        var predictions = new ArrayList<>();
        var csvFormat = CSVFormat.DEFAULT.withHeader();

        try (var parser = csvFormat.parse(in)) {
            for (var csvRow : parser) {
                var mapRow = csvRow.toMap();
                predictions.add(scoreRow(mapRow));
            }
        }
        return this.predictionsToString(predictions);
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
