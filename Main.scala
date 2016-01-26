package uk.ac.ucl.cs.mr.statnlpbook.assignment3
import collection.mutable.HashMap
/**
 * @author rockt
 */
object Main extends App {
  /**
   * Example training of a model
   *
   * Problems 2/3/4: perform a grid search over the parameters below
   */
  val learningRate = 0.01 //grid
  val vectorRegularizationStrength = 0.01 //grid
  val matrixRegularizationStrength = 0.01
  val wordDim = 10 //grid
  val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"
  
  val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  //val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)

  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }

  var lastEpochValAcc = 0.0
  def epochHook(model:Model, iter: Int, accLoss: Double): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
  }

  //StochasticGradientDescentLearner(model, trainSetName, 10, learningRate, epochHook)

  val vecRegParamas = Seq(0.1,0.2,0.3) //(0.001, 0.01, 0.05, 0.1)
  val learningRateParams = Seq(0.1)//(0.001, 0.01, 0.05, 0.1)
  val wrdDimParams = Seq(10)//(5,10,15,25)
  val ParamTriplets = collection.mutable.HashMap[(Double,Double,Int),Double]()

  for (vecRegParam <- vecRegParamas){
    for (learningRateParam <- learningRateParams) {
      for (wrdDimParam <- wrdDimParams) {
        lastEpochValAcc = 0.0
        val model: Model = new SumOfWordVectorsModel(wrdDimParam, vecRegParam)
        StochasticGradientDescentLearner(model, trainSetName, 5, learningRateParam, epochHook)
        lastEpochValAcc=100*Evaluator(model, validationSetName)
        ParamTriplets += (vecRegParam,learningRateParam,wrdDimParam) -> lastEpochValAcc
      }
    }
  }
  println(ParamTriplets)
  val (bestRegularizationParam, bestLearningRate, bestWrdDim)= ParamTriplets.maxBy(_._2)._1
  println("the best regularization parameter is %4.2f\t best learning rate is %4.2f\t best word dimension is %4d\t".format(
    bestRegularizationParam,bestLearningRate,bestWrdDim)
  )
  val bestModel: Model = new SumOfWordVectorsModel(bestWrdDim, bestRegularizationParam)
  StochasticGradientDescentLearner(bestModel, trainSetName, 5, bestLearningRate, epochHook)
  Predictor(bestModel, "test")


  /**
   * Comment this in if you want to look at trained parameters
   */
  /*
  for ((paramName, paramBlock) <- model.vectorParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }
  */
}