package com.datarobot.drum;

import com.datarobot.prediction.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.*;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Paths;
import java.sql.Time;
import java.util.*;

public class ScoringCode extends BasePredictor {
    private IPredictorInfo model;
    private boolean isRegression;
    private boolean isTimeSeries;
    private String negativeClassLabel = null;
    private String positiveClassLabel = null;
    private String[] classLabels = null;
    private String[] timeSeriesHeaders = null;
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

        try {
            if (this.isTimeSeries) {
                CSVFormat csvFormat = CSVFormat.DEFAULT.builder()
                        .setHeader(this.timeSeriesHeaders)
                        .build();
                csvPrinter = new CSVPrinter(new StringWriter(), csvFormat);
                for (var val : predictions) {
                    TimeSeriesScore<Double> value = (TimeSeriesScore<Double>) val;
                    csvPrinter.printRecord(
                            value.getSeriesId(),
                            value.getForecastTimestamp(),
                            value.getForecastPoint(),
                            value.getForecastDistance(),
                            value.getScore()
                    );
                }
            } else if (this.isRegression) {
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

    @Override
    public void configure(Map<String, Object> params) {
        this.params = params;

        String customModelPath = (String) this.params.get("__custom_model_path__");
        this.negativeClassLabel = (String) this.params.get("negativeClassLabel");
        this.positiveClassLabel = (String) this.params.get("positiveClassLabel");
        this.classLabels = (String[]) this.params.get("classLabels");

        if (this.negativeClassLabel != null && this.positiveClassLabel != null) {
            this.classLabels = new String[]{this.positiveClassLabel, this.negativeClassLabel};
        }

        this.timeSeriesHeaders = new String[]{"Id", "Timestamp", "Forecast_Point", "Forecast_Distance", "Prediction"};
        try {
            this.model = loadModel(customModelPath);
            this.isTimeSeries = this.model instanceof ITimeSeriesRegressionPredictor;
            this.isRegression = this.model instanceof IRegressionPredictor;
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String scoreReader(Reader in) throws IOException, Exception {
        var predictions = new ArrayList<>();
        var csvFormat = CSVFormat.DEFAULT.withHeader();

        try (var parser = csvFormat.parse(in)) {

            if (isTimeSeries) {
                ArrayList<Map<String, ?>> rows = new ArrayList<>();
                for (var csvRow : parser) {
                    var mapRow = csvRow.toMap();
                    rows.add(mapRow);
                }
                TimeSeriesOptions.Builder optionsBuilder = TimeSeriesOptions.newBuilder()
                        .computeIntervals(false);

                //TimeSeriesOptions options = optionsBuilder.buildForecastDateRangeRequest("2014-02-15","2014-07-16");
                TimeSeriesOptions options = optionsBuilder.buildSingleForecastPointRequest("2014-07-15");
                //List<TimeSeriesScore<Double>> result = ((ITimeSeriesRegressionPredictor) model).score(rows, options);
                List<TimeSeriesScore<Double>> result = ((ITimeSeriesRegressionPredictor) model).score(rows);
                return this.predictionsToString(result);
            } else {
                for (var csvRow : parser) {
                    var mapRow = csvRow.toMap();
                    predictions.add(scoreRow(mapRow));
                }
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
                remappedPrediction.put(this.classLabels[1], prediction.get(originalClassLabels[1]));
                remappedPrediction.put(this.classLabels[0], prediction.get(originalClassLabels[0]));
                prediction = remappedPrediction;
            }
            return prediction;
        }
    }
}
