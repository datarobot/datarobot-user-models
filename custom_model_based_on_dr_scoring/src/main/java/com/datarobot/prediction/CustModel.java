package com.datarobot.prediction;
import java.util.LinkedHashMap;
import java.util.Map;
import com.datarobot.prediction.Predictor;
import com.datarobot.prediction.Row;
import com.google.auto.service.*;

@AutoService(Predictor.class)
public class CustModel  implements Predictor{

    // list out the numeric variables in any order you
    // not necessarily matching training data
    // WARNGIN!! variable naming convention must coincide with what datarobot accepts
    // For example, in DR . are turned into _.
    @Override
    public String[] get_double_predictors() {
        String[] doubleFeatures = new String[]{"featureDouble1"};
        return doubleFeatures;
    }

    // list out the numeric variables in any order you
    // not necessarily matching training data
    // WARNGIN!! variable naming convention must coincide with what datarobot accepts
    // For example, in DR . are turned into _.
    @Override
    public String[] get_string_predictors() {
        String[] stringFeatures = new String[]{"featureStr1","featureStr2"};
        return stringFeatures;
    }
    @Override
    public double score(Row r) throws Exception {
        int numDoubleFeatures = r.d.length;
        int numStringFeatures = r.s.length;
        String[] stringFeatureNames = this.get_string_predictors();
        String[] doubleFeatureNames = this.get_double_predictors();
        for(int i = 0; i < numStringFeatures; i++) {
            System.out.printf("%s, %s\n", stringFeatureNames[i], r.s[i]);
        }
        for(int i = 0; i < numDoubleFeatures; i++) {
            System.out.printf("%s, %f\n", doubleFeatureNames[i], r.d[i]);
        }

        return 0.10101010;
    }
}
