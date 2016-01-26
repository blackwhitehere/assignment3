package uk.ac.ucl.cs.mr.statnlpbook.assignment3
import collection.mutable.MutableList
/**
 * Problem 2
 */

object StochasticGradientDescentLearner extends App {
  def apply(model: Model, corpus: String, maxEpochs: Int = 10, learningRate: Double, epochHook: (Model, Int, Double) => (Int, Double, Double, Double)): MutableList[(Int, Double, Double, Double)] = {
    val iterations = SentimentAnalysisCorpus.numExamples(corpus)
    var stats = collection.mutable.MutableList[(Int,Double,Double,Double)]()
    for (i <- 0 until maxEpochs) {
      var accLoss = 0.0
      for (j <- 0 until iterations) {
        if (j % 1000 == 0) print(s"Iter $j\r")
        val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
        //todo: update the parameters of the model and accumulate the loss
        val ls = model.loss(sentence, target)
        accLoss += ls.forward()
        ls.backward()
        ls.update(learningRate)
      }
      val t = epochHook(model, i, accLoss)
      stats += t
    }
    stats
  }
}

