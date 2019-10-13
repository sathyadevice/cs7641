package opt.sconjeevaram;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import func.nn.activation.*;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;
import java.util.StringTokenizer;
public class diabetes_sa {
    private static Instance[] instances = initializeInstances();
    private static Instance[] train_set = Arrays.copyOfRange(instances, 0, 538);
    private static Instance[] test_set = Arrays.copyOfRange(instances, 538, 769);

    private static DataSet set = new DataSet(train_set);

    private static int inputLayer = 50, hiddenLayer=20, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"SA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");



    public static void write_output_to_file(String output_dir, String file_name, String results, boolean final_result) {
        try {
            if (final_result) {
                String augmented_output_dir = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date());
                String full_path = augmented_output_dir + "/" + file_name;
                Path p = Paths.get(full_path);
                if (Files.notExists(p)) {
                    Files.createDirectories(p.getParent());
                }
                PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                synchronized (pwtr) {
                    pwtr.println(results);
                    pwtr.close();
                }
            }
            else {
                String full_path = output_dir + "/" + new SimpleDateFormat("yyyy-MM-dd").format(new Date()) + "/" + file_name;
                Path p = Paths.get(full_path);
                Files.createDirectories(p.getParent());
                Files.write(p, results.getBytes());
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }

    }



    public static void main(String[] args) {

        String final_result = "";


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        int[] iterations = {10, 100, 500, 1000, 2500, 5000};

        double[] coolings = {0.15,0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95};

        for (int trainingIterations : iterations) {
            results = "";
            for (double cooling : coolings) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                oa[0] = new SimulatedAnnealing(1E11, cooling, nnop[0]);
                train(oa[0], networks[0], oaNames[0], trainingIterations); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[0].getOptimal();
                networks[0].setWeights(optimalInstance.getData());

                // Calculate Training Set Statistics //
                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < train_set.length; j++) {
                    networks[0].setInputValues(train_set[j].getData());
                    networks[0].run();
                    
                    actual = Double.parseDouble(train_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[0].getOutputValues().toString());

                    //System.out.println("actual is " + actual);
                    //System.out.println("predicted is " + predicted);

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTrain Results for SA:" + cooling + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\nIterations: " + df.format(trainingIterations) + "\n";

//                final_result = oaNames[0] + "," + trainingIterations + "," + cooling + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
//                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
//                        "," + df.format(testingTime);
//                write_output_to_file("..\\ABAGAIL\\src\\opt\\sconjeevaram\\Optimization_Results", "diabetes_results_sa.csv", final_result, true);

                // Calculate Test Set Statistics //
                start = System.nanoTime();
                correct = 0;
                incorrect = 0;
                for (int j = 0; j < test_set.length - 1; j++) {
                    networks[0].setInputValues(test_set[j].getData());
                    networks[0].run();

                    actual = Double.parseDouble(test_set[j].getLabel().toString());
                    predicted = Double.parseDouble(networks[0].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nTest Results for SA: " + cooling + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\nIterations: " + df.format(trainingIterations) + "\n";

                final_result = oaNames[0] + "," + trainingIterations + "," + cooling + "," + "training accuracy" + "," + df.format(correct / (correct + incorrect) * 100) 
                		+ "," + "testing accuracy" + "," + df.format(correct / (correct + incorrect) * 100)
                        + "," + "training time" + "," + df.format(trainingTime) + "," + "testing time" +
                        "," + df.format(testingTime);
                write_output_to_file("..\\ABAGAIL\\src\\opt\\sconjeevaram\\Optimization_Results", "diabetes_results_sa.csv", final_result, true);
            }
            System.out.println("results for iteration: " + trainingIterations + "---------------------------");
            System.out.println(results);
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iteration) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");
        int trainingIterations = iteration;
        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double train_error = 0;
            for(int j = 0; j < train_set.length; j++) {
                network.setInputValues(train_set[j].getData());
                network.run();

                Instance output = train_set[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                train_error += measure.value(output, example);
            }


            //System.out.println("training error :" + df.format(train_error)+", testing error: "+df.format(test_error));
        }
    }

    private static Instance[] initializeInstances() {

    	ArrayList<Double[]> listOfEnteries = null;

        try {
        	String filePath = "..\\ABAGAIL\\src\\opt\\sconjeevaram\\diabetes.csv";
        	BufferedReader br = new BufferedReader(new FileReader(new File(filePath)));
        	listOfEnteries = new ArrayList<>();
            br.readLine();
            String entry = "";
            int lineNumber = 1;
            while ((entry = br.readLine()) != null) {
                Double[] entryarray = new Double[9];
                StringTokenizer tokenizer = new StringTokenizer(entry, ",");
                int counter = 0;
                while (tokenizer.hasMoreTokens()) {
                    try {
                        entryarray[counter] = Double.parseDouble(tokenizer.nextToken());
                        counter++;
                    } catch (NumberFormatException ef) {
                        System.out.println("LineNUmber : " + lineNumber + ", entryNumber : " + counter + " is not a number");
                        entryarray[counter] = 0.0;
                        counter++;
                    }
                }
                listOfEnteries.add(entryarray);
                lineNumber++;
            }
            br.close();  
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        // Accessing the Data
        if (listOfEnteries != null) {
            for (int line = 0; line < listOfEnteries.size(); line++) {
                Double[] arr = listOfEnteries.get(line);
                for (int entry = 0; entry < 9; entry++) {
                    System.out.print(arr[entry] + ",");
                }
                System.out.println("");
            }
        } else {
            System.out.println("Data not able to fetch -- possible file not found");
        }
        
        Instance[] instances = new Instance[listOfEnteries.size()];
        int line = 0;
        for(int i = 0; i < instances.length; i++, line++) {
        	Double[] arr = listOfEnteries.get(line);
            instances[i] = new Instance(arr.length);
            //System.out.println(arr[8].intValue());
            instances[i].setLabel(new Instance(arr[8].intValue()< 0 ? 0 : 1));
        }
        
        return instances;
    }
}