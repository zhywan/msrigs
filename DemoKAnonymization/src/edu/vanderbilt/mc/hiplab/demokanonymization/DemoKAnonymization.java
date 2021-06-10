package edu.vanderbilt.mc.hiplab.demokanonymization;
//***************************************************************
// Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
// Component: MSRIGS K-Anonymization baseline using ARX API
// Copyright 2021-2021 Zhiyu Wan, HIPLAB, Vanderilt University
// Compatible with openjdk 16. Package dependencies: libarx-3.9.0
//***************************************************************
import org.deidentifier.arx.*;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.exceptions.RollbackRequiredException;
import org.deidentifier.arx.metric.Metric;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Iterator;
import java.util.Arrays;

public class DemoKAnonymization {
    public static void main(String[] args) throws IOException {
        int n_iter = 100;  // number of iterations
        String exp_id = "2058";
        String data_dir = "exp/" + exp_id + "/data/";
        String output_dir = "exp/" + exp_id + "/output_strategy/";
        for (int ii = 0; ii < n_iter; ii++) {
            // Define data
            System.out.println(exp_id + "_i" + (ii + 1));
            Data data = Data.create(data_dir + "target_data/i" + ii + ".csv", Charset.defaultCharset(), ',');
            DataHandle handle = data.getHandle();
            String[] attribute_name = new String[14];
            for (int i = 0; i < 14; i++) {
                attribute_name[i] = handle.getAttributeName(i);
            }
            // input weights
            double[] weights = new double[14];
            try (BufferedReader br = new BufferedReader(new FileReader(data_dir + "weighted_entropy/i" + ii + ".csv"))) {
                String line;
                int jj = 0;
                while ((line = br.readLine()) != null) {
                    weights[jj] = Double.parseDouble(line);
                    jj = jj + 1;
                }
            }

            for (int i = 0; i < 14; i++) {
                // Define hierarchies
                AttributeType.Hierarchy hierarchy = AttributeType.Hierarchy.create(data_dir + "hierarchy/hierarchy_" + attribute_name[i] + ".csv", Charset.defaultCharset(), ',');
                // Set hierarchies
                data.getDefinition().setAttributeType(attribute_name[i], hierarchy);
                // Set data types
                data.getDefinition().setDataType(attribute_name[i], DataType.DECIMAL);
                // Define quasi-identifying attribute
                data.getDefinition().setAttributeType(attribute_name[i], AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
            }

            // Create an instance of the anonymizer
            ARXAnonymizer anonymizer = new ARXAnonymizer();

            // Creat a configuration
            ARXConfiguration config = ARXConfiguration.create();
            // Define privacy model
            config.addPrivacyModel(new KAnonymity(2));
            config.setSuppressionLimit(1d); // favors suppression over generalization
            for (int i = 0; i < 14; i++) {
                config.setAttributeWeight(attribute_name[i], weights[i]);
            }

            // Metric for the information loss
            config.setQualityModel(Metric.createPrecisionMetric(0d, Metric.AggregateFunction.valueOf("GEOMETRIC_MEAN")));  // favors suppression over generalization

            // Compute the result
            ARXResult result = anonymizer.anonymize(data, config);

            // Get result of global recoding
            DataHandle optimum = result.getOutput();
            try {
                // Now apply local recoding to the result
                result.optimizeIterativeFast(optimum, 1d / handle.getNumRows());
            } catch (RollbackRequiredException e) {
                // This part is important to ensure that privacy is preserved, even in case of exceptions
                optimum = result.getOutput();
            }

            Iterator<String[]> transformed = optimum.iterator();
            transformed.next();
            try {
                FileWriter writer = new FileWriter(output_dir + "/i" + ii + ".csv", false);
                BufferedWriter bufferedWriter = new BufferedWriter(writer);
                while (transformed.hasNext()) {
                    String line = Arrays.toString(transformed.next());
                    String[] cells = line.split(", ");
                    for (String s: cells){
                        if (s.contains("*")){
                            bufferedWriter.write("0");
                        }
                        else{
                            bufferedWriter.write("1");
                        }
                        if (s.contains("]")){
                            bufferedWriter.write("\n");
                        }
                        else{
                            bufferedWriter.write(",");
                        }
                    }
                }
                bufferedWriter.close();
            } catch (IOException e){
                e.printStackTrace();
            }
        }
    }
}
