import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.M5P;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import javax.sound.sampled.Line;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class EnergyLoad {
    Instances data;
    public void loadData(String arg){
        CSVLoader loader = new CSVLoader();
        try {
            loader.setSource(new File(arg));
            data = loader.getDataSet();
 //           System.out.print(data);
        }catch(IOException ex) {
            System.err.println("Could not load data.");
        }
    }
    public void toFilter(){
        try {
            Remove remove = new Remove();
            remove.setOptions(new String[]{"-R", data.numAttributes() + ""});
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            data.setClassIndex(data.numAttributes()-1);
        } catch (Exception ex){
            System.err.println("Failed in filtering");
        }
    }
    public void pieceWiseLinear(){
        M5P md5 = new M5P();
        try {
            md5.setOptions(new String[]{""});
            md5.buildClassifier(data);
            System.out.println(md5);
        } catch (Exception ex){
            System.err.println("unable to apply the piecewise Linear Regression.");
        }
    }
    public void linearRegression() {
        try {
            LinearRegression model = new LinearRegression();
            model.buildClassifier(data);
            System.out.println(model);
            Evaluation eval_roc = new Evaluation(data);
            eval_roc.crossValidateModel(model, data , 10, new Random(1), new Object[]{});
            System.out.println(eval_roc.toSummaryString());
        } catch(Exception ex){
            System.err.print("Linear regression has failed");
        }
    }
    public static void main(String[] args) {
        EnergyLoad eload = new EnergyLoad();
        eload.loadData("/home/brambila/IdeaProjects/MLWJ/src/main/resources/ENB2012_data.csv");
        eload.toFilter();
        eload.pieceWiseLinear();
//        eload.linearRegression();
    }
}
