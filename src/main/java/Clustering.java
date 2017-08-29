import weka.clusterers.EM;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

public class Clustering {
    Instances data;
    public void loadData(String file){
        try {
            data = new Instances(new BufferedReader(new FileReader(file)));
        } catch (FileNotFoundException){
            System.err.println("File not found.");
        }
    }
    public void applyClustering(){
        EM model = new EM();
        model.buildClusterer(data);
    }
}
