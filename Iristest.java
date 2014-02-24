package com.iris.data.mahout;


import java.io.FileInputStream;

import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Iterator;







//import org.apache.commons.math.util.OpenIntToDoubleHashMap.Iterator;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Resources;
//import javax.annotation.Resources;

public class Iristest
{
	List<Integer> train;
	List<Integer> test;
	ArrayList <Integer>target_train,target_test;
	ArrayList<DenseVector> data_train,data_test;
	ArrayList<Integer> order_train,order_test;
	List<String> raw_train,raw_test;
	Random random;
	

    public  void buildTrainData() throws IOException {
	 
	  // Snip ...
	  Splitter onComma = Splitter.on(",");
	 
	  // read the data
	  raw_train = Resources.readLines(Resources.getResource("iris-2D-mod.csv"), Charsets.UTF_8);
	  
	   //System.out.println(raw_train);
	  // holds features
	  data_train = Lists.newArrayList();
	 
	  // holds target variable
	  target_train = Lists.newArrayList();
	 
	  // for decoding target values
	  Dictionary dict = new Dictionary();
	 
	  // for permuting data later
	  order_train = Lists.newArrayList();
	  for (String line : raw_train.subList(1, raw_train.size())) {
	    // ; gets a list of indexes
	    order_train.add(order_train.size());
	 
	    // parse the predictor variables
	    DenseVector v = new DenseVector(3);
	    DenseVector v_withOriginalIntercept=new DenseVector(3);
	     
	    v.set(0, 1);
	    int i = 1;
	    Iterable<String> values = onComma.split(line);
	    
	    for (String value : Iterables.limit(values, 2)) {
	      v.set(i++, Double.parseDouble(value));
	    	
	    }
	    v_withOriginalIntercept=(DenseVector) v.normalize();
	    v_withOriginalIntercept.set(0, 1);
	    data_train.add(v);
	 
	    // and the target
	    target_train.add(dict.intern(Iterables.get(values, 2)));
	  }
	  
}

public static void main(String[] args) throws Exception {
	
	Iristest t1=new Iristest();
	t1.buildTrainData();
	t1.trainable();

}
public void trainable() throws Exception {
	
	
	double heldOutPercentage = 0.10;
	int cutoff=(int)(heldOutPercentage*order_train.size());
    Collections.shuffle(order_train);
    List<Integer> test =  order_train.subList(0, cutoff);
    List<Integer> train  =  order_train.subList(cutoff, order_train.size());
    

double accuracy=0.0,temp=0.0;;

AdaptiveLogisticRegression lr = new AdaptiveLogisticRegression(3, 3, new L1());
for(Integer k:train)
{
	lr.train(target_train.get(k),data_train.get(k));
}

 lr.close();
 ModelSerializer.writeBinary("/home/psuryawanshi/Downloads/alr2.model", lr.getBest().getPayload().getLearner());//.getModels().get(0));
 InputStream in=new FileInputStream("/home/psuryawanshi/Downloads/alr2.model");
 CrossFoldLearner best=ModelSerializer.readBinary(in,CrossFoldLearner.class);
 System.out.println("auc="+best.auc()+"percentCorrect="+best.percentCorrect()+"LogLikelihood="+best.getLogLikelihood());
  
  in.close();
  // check the accuracy on held out data
	int x=0,r=0;
 	 for (Integer k : test) {
	//Test testProfile;
		  
		  r=best.classifyFull(data_train.get(k)).maxValueIndex();
		  //System.out.println("r="+r+" "+"target_train.get(k)="+target_train.get(k));
		  x += r == target_train.get(k) ? 1 : 0;
	  }
    
 	accuracy=(double)(x)/(double)(test.size());
  
System.out.printf("accuracy is %f\n", accuracy);
}
}
