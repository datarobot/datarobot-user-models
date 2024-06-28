package com.mlpiper.components.test_java_connected_multiple_jars;

import com.mlpiper.MCenterComponent;
import com.mlpiper.components.test_java_connected_multiple_jars.JavaConnectedAddition;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

public class JavaConnectedComponent extends MCenterComponent
{
    @Override
    public List<Object> materialize(List<Object> parentDataObjects) throws Exception {
        System.out.println("Running JavaConnectedComponent");
        JavaConnectedAddition.printHello();

        List<Object> outputs = new ArrayList<>();
        return outputs;
    }
}
