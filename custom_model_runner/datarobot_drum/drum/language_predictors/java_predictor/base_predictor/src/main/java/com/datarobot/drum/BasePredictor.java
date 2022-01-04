package com.datarobot.drum;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

public abstract class BasePredictor {
    protected String name;

    public BasePredictor(String name) {
        this.name = name;
    }

    public BasePredictor setName(String name) {
        this.name = name;
        return this;
    }

    /**
    * Called once to configure predictor.
    * E.g. use it to load a model.
    * @param params Configuration parameters passed from DRUM.
    */
    public abstract void configure(Map<String, Object> params) throws Exception;

    /**
    * Make predictions on input scoring data.
    * @param inputBytes Input data as binary.
    * @return predictions as CSV string.
    */
    public abstract String predict(byte[] inputBytes) throws Exception;


    /**
    * Make predictions on input scoring data.
    * @param inputBytes Input data as binary -> byte[].
    * @return can technically return anything but DRUM expects either String or byte[].
    */
    public abstract <T> T predictUnstructured(byte[] inputBytes, String mimetype, String charset, Map<String, String> query) throws Exception;


    /**
    * Make predictions on input CSV. This method is used in java_predictor.py, when
    * input data is larger than 33K and data is temporary saved into a CSV.
    * @param inputFilename Input data as a temporary CSV file.
    * @return predictions as CSV string.
    */
    public String predictCSV(String inputFilename) throws Exception {
        String ret = null;
        try {
            byte[] inputBytes = Files.readAllBytes(Paths.get(inputFilename));
            ret = this.predict(inputBytes);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }
}
