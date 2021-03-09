package com.datarobot.drum;

import java.io.IOException;

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
    * Make predictions on input scoring data.
    * @param inputBytes Input data as binary.
    * @return predictions as CSV string.
    */
    public abstract String predict(byte[] inputBytes) throws Exception;

    /**
    * Make predictions on input CSV.
    * @param inputFilename Input data as a temporary CSV file.
    * @return predictions as CSV string.
    */
    public abstract String predictCSV(String inputFilename) throws Exception;
}
