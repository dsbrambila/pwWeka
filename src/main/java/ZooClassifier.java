import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.util.Random;


public class ZooClassifier{
    private Instances data;
    public void getData(String filePath) {
        try {
            DataSource source = new DataSource(filePath);
            data = source.getDataSet();
            System.out.println(data.numInstances() + " instances loaded.");
        }catch (Exception ex){
            System.err.println("Failed to load data ");
        }
    }
    public void filtering(String[] opts) {
        try {
            Remove remove = new Remove();
            remove.setOptions(opts);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
        }
        catch (Exception ex){
            System.err.println("Filtering failed.");
        }
    }
    public void FS() {
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        AttributeSelection attSelect = new AttributeSelection();
        attSelect.setEvaluator(eval);
        attSelect.setSearch(search);
        try {
            attSelect.SelectAttributes(data);
            int[] indices = attSelect.selectedAttributes();
            System.out.println(Utils.arrayToString(indices));
        } catch (Exception ex){
            System.err.println("Failed in attribute selection.");
        }
    }
    public void model(String[] opts) {
        J48 tree = new J48();
        Classifier cl = new J48();
        try {
            tree.setOptions(opts);
            tree.buildClassifier(data);
            Evaluation eval_roc = new Evaluation(data);
            eval_roc.crossValidateModel(cl, data , 10, new Random(1), new Object[]{});
            System.out.println(eval_roc.toSummaryString());
            confusionMatrix(eval_roc);
 //           tview(tree);
        } catch(Exception ex){
            System.err.println("Could not create model.");
        }
    }
    private void tview(J48 tree) {
        try {
            TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
            JFrame frame = new javax.swing.JFrame("Tree Visualizer");
            frame.setSize(800,500);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.getContentPane().add(tv);
            frame.setVisible(true);
            tv.fitToScreen();
        }
        catch (Exception ex){
            System.err.println("Could not create visualization");
        }

    }
    public void confusionMatrix(Evaluation eval_roc){
        try {
            double [] [] confusionMatrix = eval_roc.confusionMatrix();
            System.out.print(eval_roc.toMatrixString());
        } catch (Exception ex) {
            System.err.println("Could not eval confusion matrix");
        }
    }
    public static void main(String[] args) {
        ZooClassifier classification = new ZooClassifier();

        classification.getData("src/main/resources/zoo.arff");
        classification.filtering(new String[]{"-R","1"});
        classification.FS();
        classification.model(new String[]{"-U"});
   //     classification.confusionMatrix();

    }
}
