/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package netflix;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.misc.SerializedClassifier;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author Carolina
 */
public class redNeuronal {
    
    public void entrenamiento(String parametros){
        
        try {
            FileReader trainreader = new FileReader("src/archivos_arff/netflixprueba.arff");
            Instances train = new Instances(trainreader);
			train.setClassIndex(train.numAttributes()-1);
                        
             MultilayerPerceptron mlp = new MultilayerPerceptron();
			mlp.setOptions(Utils.splitOptions(parametros));
			mlp.buildClassifier(train); //construir clasificador   
                        
            Debug.saveToFile("src/red/RedNeuronalEntrenada.train", mlp);
			
			//evaluar el entrenamiento
			Evaluation evaluarTrain = new Evaluation(train);
			evaluarTrain.evaluateModel(mlp, train);
			System.out.println(evaluarTrain.toSummaryString("Resumen", false));
			System.out.println(evaluarTrain.toMatrixString("Matrix Con."));
			
			trainreader.close();
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    
    public void testeo(){
		
		try {
			FileReader testReader = new FileReader("src/archivos_arff/testeoNetflix.arff");
			Instances test = new Instances(testReader);
			test.setClassIndex(test.numAttributes()-1);
			
			Evaluation evalTest=new Evaluation(test);
			
			//fijar modelo de red neuronal serializado
			
			SerializedClassifier classifier = new SerializedClassifier();
			classifier.setModelFile(new File ("src/red/RedNeuronalEntrenada.train"));
			Classifier mlp = classifier.getCurrentModel(); //obtengo el modelo y lo igualo 
			
			evalTest.evaluateModel(mlp, test);
			System.out.println(evalTest.toSummaryString("Resumen", false));
			System.out.println(evalTest.toMatrixString("Matrix Con."));
			
			testReader.close();
			
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    
     public void prediccion(){
		
		FileReader predReader;
		try {
			predReader = new FileReader("src/archivos_arff/diabetesPred.arff");
			Instances pred = new Instances(predReader);
			pred.setClassIndex(pred.numAttributes()-1);
			
			Evaluation evalPred=new Evaluation(pred);
			
			SerializedClassifier classifier = new SerializedClassifier();
			classifier.setModelFile(new File ("src/red/RedNeuronalEntrenada.train"));
			
			Classifier mlp = classifier.getCurrentModel(); //obtengo el modelo y lo igualo 
			evalPred.evaluateModel(mlp, pred);
			
			ArrayList<Prediction> predicciones = evalPred.predictions();
			for (int i = 0; i < predicciones.size(); i++) {
				Prediction p = predicciones.get(i);
				System.out.println(p.predicted());
			}

            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(redNeuronal.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    
    
}
