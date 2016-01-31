package uk.ac.ucl.cs.mr.statnlpbook.assignment3
import scala.util.Random

/**
 * Created by stan on 29-Jan-16.
 */
object GridSearchSOW extends App {
  def epochHook(model:Model, iter: Int, accLoss: Double): (Int,Double,Double,Double) = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
    val rtn = (iter,accLoss,100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName))
    rtn
  }

  val trainSetName = "train"
  val validationSetName = "dev"
  var lastEpochValAcc = 0.0


  val vecRegParams = Seq(0.05,0.1,0.15)
  val learningRateParams = Seq(0.02,0.01,0.005)
  val wrdDimParams = Seq(10,15,20)
  val ParamTriplets = collection.mutable.HashMap[(Double,Double,Int),Double]()
  for (vecRegParam <- vecRegParams){
    for (learningRateParam <- learningRateParams) {
      for (wrdDimParam <- wrdDimParams) {
        val model: Model = new SumOfWordVectorsModel(wrdDimParam, vecRegParam)
        val stats=StochasticGradientDescentLearner(model, trainSetName, 20, learningRateParam, epochHook)
        ParamTriplets += (vecRegParam,learningRateParam,wrdDimParam) -> stats.last._4
      }
    }
  }
  //println(ParamTriplets)
  val (bestRegularizationParam, bestLearningRate, bestWrdDim)= ParamTriplets.maxBy(_._2)._1
  println("the best regularization parameter is %4.2f\t best learning rate is %4.2f\t best word dimension is %4d\t".format(
    bestRegularizationParam,bestLearningRate,bestWrdDim)
  )
  val bestModel: Model = new SumOfWordVectorsModel(bestWrdDim, bestRegularizationParam) //best steup - 15,0.15
  val listOfStats=StochasticGradientDescentLearner(bestModel, trainSetName, 40, bestLearningRate, epochHook) //best setup -0.005

  val statsString=listOfStats.mkString("\n")
  scala.tools.nsc.io.File("stats.csv").writeAll(statsString)

  Predictor(bestModel, "test")

  CleanAndWrite(bestModel)
}
