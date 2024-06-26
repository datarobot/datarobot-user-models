package com.mlpiper.components.test_java_predict_middle;

import com.mlpiper.MCenterComponent;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import javax.management.ObjectName;
import javax.management.MBeanServer;
import javax.management.MalformedObjectNameException;
import javax.management.MBeanException;
import javax.management.AttributeNotFoundException;


/**
 * Hello world!
 *
 */
public class JavaPredictMiddleComponent extends MCenterComponent
{
    @Override
    public List<Object> materialize(List<Object> parentDataObjects) throws Exception {
        System.out.println("JavaPredictMiddleComponent - materialize");

        int exitValue = (int) params.getOrDefault ("exit_value", 0);
        System.out.println("param - exitValue:    " + exitValue);

        int iter = (int) params.getOrDefault("iter", 5);
        System.out.println("param - iter:         " + iter);

        String modelPathStr = (String) params.get("input_model");
        System.out.println("param - modelPath:    " + modelPathStr);

        int memAllocSizeMB = (int) params.getOrDefault("mem-size-alloc-step", 100);
        System.out.println("param - memAllocSizeMB:    " + memAllocSizeMB);

        String inputStr = null;
        if (parentDataObjects.size() > 0) {
            inputStr = (String) parentDataObjects.get(0);
            System.out.println("input - 0:            " + inputStr);
        } else {
            inputStr = "";
            System.out.println("Parent data object list is empty!");
        }

        Paths.get(modelPathStr);
        Path modelFile = Paths.get(modelPathStr);
        if (!modelFile.toFile().exists()) {
            throw new Exception("Model file does exits");
        }

        System.out.println();
        System.out.println("Loop:");

        List<int[]> dataCollection = new ArrayList<>();
        for (int i=0 ; i < iter ; i++) {
            System.out.println(String.format("Iter: %d/%d", i, iter));
            TimeUnit.SECONDS.sleep(1);

            consumeMemory(dataCollection, memAllocSizeMB, i);
        }
        System.out.println("Data collection size: " + dataCollection.size());

        System.out.println("About to exit with: " + exitValue);
        if (exitValue > 0) {
            System.exit(exitValue);
        } else if (exitValue < 0) {
            throw new Exception("Exception raised by java code - test component");
        }

        List<Object> outputs = new ArrayList<>();
        outputs.add(inputStr);
        return outputs;
    }

    private void consumeMemory(List<int[]> dataCollection, int memAllocSizeMB, int cycle) {
        long memAllocSizeBytes = ((long)memAllocSizeMB * 1024 * 1024);

        int[] data = null;
        while (memAllocSizeBytes > 0) {
            long memAllocSizeInts = memAllocSizeBytes / 4 /*sizeof(int)*/;
            if (memAllocSizeInts > Integer.MAX_VALUE) {
                memAllocSizeBytes -= (Integer.MAX_VALUE * 4);
                data = new int[Integer.MAX_VALUE];
            } else {
                memAllocSizeBytes = 0;
                data = new int[(int)memAllocSizeInts];
            }
            Arrays.fill(data, cycle);
            dataCollection.add(data);
        }

        System.out.println("Total allocated memory (MB): " + dataCollection.size() * memAllocSizeMB);

        // printMemoryInfo();
    }

    private void printMemoryInfo() {
        MemoryUsage heapMemoryUsage = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        System.out.println("JVM memory: " + heapMemoryUsage.toString());

        try {
            MBeanServer mBeanServer = ManagementFactory.getPlatformMBeanServer();
            Object freePhysMemObj = mBeanServer.getAttribute(new ObjectName("java.lang","type","OperatingSystem"),
                                                      "FreePhysicalMemorySize");
            double freePhysMem = Long.parseLong(freePhysMemObj.toString()) / 1024.0 / 1024.0 / 1024.0;
            System.out.println("Host free memory: " + freePhysMem + " GB");
        } catch (Exception ex) {
            System.out.println(ex);
        }
    }

    public static void main(String[] args ) throws Exception {
        JavaPredictMiddleComponent middleComponent = new JavaPredictMiddleComponent();
        ArgumentParser parser = ArgumentParsers.newFor("Checksum").build()
                                               .defaultHelp(true)
                                               .description("Calculate checksum of given files.");
        parser.addArgument("--exit-value")
              .type(Integer.class)
              .setDefault(0)
              .help("Exit status of component");
        parser.addArgument("--iter")
              .type(Integer.class)
              .setDefault(5)
              .help("Number of 1 second iterations to perform");

        parser.addArgument("--input-model")
              .help("Path to input model to consume");

        Namespace options = null;
        try {
            options = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        System.out.println(options);

        middleComponent.configure(options.getAttrs());
        List<Object> parentObjs = new ArrayList<Object>();
        parentObjs.add("just-a-string");
        List<Object> outputs = middleComponent.materialize(parentObjs);
        for (Object obj: outputs) {
            System.out.println("Output: " + obj.toString());
        }
    }
}
