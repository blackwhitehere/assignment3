package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import collection.mutable.MutableList

/**
 * Created by stan on 26-Jan-16.
 * This is a new object to apply trained model on test set and saved results in required format
 */
object Predictor {
  def apply(model:Model, corpus:String): Unit ={
    val total = SentimentAnalysisCorpus.numExamples(corpus)
    val myCorpus = SentimentAnalysisCorpus.getCorpus(corpus)
    val predictions= MutableList[String]()
    for (i <- 0 until total) {
      val (sentence, target) = myCorpus(i)
      val predict = model.predict(sentence)
      if (predict) predictions += "1"
      else predictions += "0"
    }
    val string=predictions.mkString("\n")
    scala.tools.nsc.io.File("predictions.txt").writeAll(string)
  }
}
