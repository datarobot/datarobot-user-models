package com.datarobot.drum;

import com.datarobot.prediction.Explanation;
import com.datarobot.prediction.IClassificationPredictor;
import com.datarobot.prediction.IPredictorInfo;
import com.datarobot.prediction.IRegressionPredictor;
import com.datarobot.prediction.Predictors;
import com.datarobot.prediction.Score;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;
import java.util.*;

public class ScoringCode extends BasePredictor {
    private IPredictorInfo model;
    private boolean isRegression;
    private boolean withExplanations = false;
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
            if (file.isFile() && file.getName().toLowerCase().endsWith(".jar")) {
                urls.add(new URL("file://" + file.getCanonicalPath()));
            }
        }
        var urlClassLoader = new URLClassLoader(urls.toArray(new URL[urls.size()]));
        return Predictors.getPredictor(urlClassLoader);
    }

    @Override
    public String predict(byte[] inputBytes) throws Exception {
        String ret = null;

        try {
            ret = this.scoreReader(new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes))));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    @Override
    public <T> T predictUnstructured(byte[] inputBytes, String mimetype, String charset, Map<String, String> query) throws Exception {
        throw new Exception("Not Implemented");
    }
    
    private String predictionsToString(List<?> predictions) throws Exception {
        CSVPrinter csvPrinter = null;

        ArrayList<String> explanationsHeaders = null;
        if (this.withExplanations) {
            if (predictions.size() > 0) {
                explanationsHeaders = new ArrayList<String>();
                var scoreObj = (Score)predictions.get(0);
                var explList = scoreObj.getPredictionExplanation();
                for (int i = 1; i <= explList.size(); i++) {
                    var prefix = "explanation_" + String.valueOf(i);
                    explanationsHeaders.add(prefix + "_feature_name");
                    explanationsHeaders.add(prefix + "_strength");
                    explanationsHeaders.add(prefix + "_actual_value");
                    explanationsHeaders.add(prefix + "_qualitative_strength");
                }
            }
        }

        try {
            var headers = new ArrayList<String>();
            if (this.isRegression) {
                headers.add("Predictions");
                if (this.withExplanations) {
                    headers.addAll(explanationsHeaders);
                    csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(headers.toArray(String[]::new)));
                    for (var value : predictions) {
                        var scoreObj = (Score)value;
                        var score = scoreObj.getScore();
                        List<Explanation> explList = scoreObj.getPredictionExplanation();
                        var values = new ArrayList<>();
                        values.add(score);
                        for (Explanation expl : explList) {
                            values.add(expl.getFeatureName());
                            values.add(expl.getStrength());
                            values.add(expl.getFeatureValue());
                            values.add(expl.getStrengthScore());
                        }
                        csvPrinter.printRecord(values);
                    }
                } else {
                    csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(headers.toArray(String[]::new)));
                    for (var value : predictions) {
                        csvPrinter.printRecord(value);
                    }
                }
            } else {
                headers.addAll(Arrays.asList(this.classLabels));
                if (this.withExplanations) {
                    headers.addAll(explanationsHeaders);
                    csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(headers.toArray(String[]::new)));

                    for (var value : predictions) {
                        var predRow = new ArrayList<>();
                        var scoreObj = (Score)value;

                        HashMap<String, Double> score = (HashMap<String, Double>) scoreObj.getScore();
                        var scores = Arrays.stream(this.classLabels).map(score::get).toArray(Double[]::new);
                        predRow.addAll(Arrays.asList(scores));

                        List<Explanation> explList = scoreObj.getPredictionExplanation();
                        for (Explanation expl : explList) {
                            predRow.add(expl.getFeatureName());
                            predRow.add(expl.getStrength());
                            predRow.add(expl.getFeatureValue());
                            predRow.add(expl.getStrengthScore());
                        }
                        csvPrinter.printRecord(predRow);
                    }
                } else {
                    csvPrinter = new CSVPrinter(new StringWriter(), CSVFormat.DEFAULT.withHeader(headers.toArray(String[]::new)));
                    for(var val : predictions) {
                        HashMap<String, Double> value = (HashMap<String, Double>) val;
                        var predRow = Arrays.stream(this.classLabels).map(value::get).toArray(Double[]::new);
                        csvPrinter.printRecord(Arrays.asList(predRow));
                    }
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
    public void configure(Map<String, Object> params) {
        this.params = params;

        String customModelPath = (String) this.params.get("__custom_model_path__");
        this.negativeClassLabel = (String) this.params.get("negativeClassLabel");
        this.positiveClassLabel = (String) this.params.get("positiveClassLabel");
        this.classLabels = (String[]) this.params.get("classLabels");
        this.withExplanations = (boolean) this.params.getOrDefault("withExplanations", false);

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
        Object scoreObj = null;
        if (isRegression) {
            if (withExplanations) {
                scoreObj = (Object)((IRegressionPredictor) model).scoreWithExplanations(row);
            } else {
                scoreObj = ((IRegressionPredictor) model).score(row);
            }
        } else {
            Map<String, Double> scoreValueRef = null;
            if (withExplanations) {
                var sc = ((IClassificationPredictor) model).scoreWithExplanations(row);
                scoreValueRef = sc.getScore();
                scoreObj = (Object)sc;
            } else {
                scoreObj = ((IClassificationPredictor) model).score(row);
                scoreValueRef = (Map<String, Double>)scoreObj;
            }

            var originalClassLabels = ((IClassificationPredictor) model).getClassLabels();

            if (originalClassLabels.length == 2) {
                scoreValueRef.put(this.classLabels[1], scoreValueRef.remove(originalClassLabels[1]));
                scoreValueRef.put(this.classLabels[0], scoreValueRef.remove(originalClassLabels[0]));
            }
        }
        return scoreObj;
    }
}
