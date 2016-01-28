package uk.ac.ucl.cs.mr.statnlpbook.assignment3
import collection.mutable.HashMap
import scala.util.matching.Regex
import scala.util.Random
//import scalax.io.
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
  //changed to evaluate on a given model
  def epochHook(model:Model, iter: Int, accLoss: Double): (Int,Double,Double,Double) = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
    val rtn = (iter,accLoss,100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName))
    rtn
  }

  //StochasticGradientDescentLearner(model, trainSetName, 10, learningRate, epochHook)

  val vecRegParamas = Seq(0.1,0.2,0.3)
  val learningRateParams = Seq(0.01,0.005,0.002)
  val wrdDimParams = Seq(10,15,20)
  val ParamTriplets = collection.mutable.HashMap[(Double,Double,Int),Double]()
  for (vecRegParam <- vecRegParamas){
    for (learningRateParam <- learningRateParams) {
      for (wrdDimParam <- wrdDimParams) {
        lastEpochValAcc = 0.0
        val model: Model = new SumOfWordVectorsModel(wrdDimParam, vecRegParam)
        StochasticGradientDescentLearner(model, trainSetName, 20, learningRateParam, epochHook)
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
  val bestModel: Model = new SumOfWordVectorsModel(bestWrdDim, bestRegularizationParam) //best steup - 15,0.15
  val listOfStats=StochasticGradientDescentLearner(bestModel, trainSetName, 40, bestLearningRate, epochHook) //best setup -0.005
  Predictor(bestModel, "test")
  val statsString=listOfStats.mkString("\n")

  //var wordVectors =bestModel.vectorParams.values.map(_.param)
  //val wordLabels=bestModel.vectorParams.keys.mkString("\n")
  //val wordSentiment =bestModel.vectorParams.values.map(a => bestModel.scoreSentence(a).forward())
  //val discreteWordSentiment = wordSentiment.map(a=>(10*a).toInt).mkString("\n")


  val random=new Random()
  val sample=random.shuffle(bestModel.vectorParams).take(5000)
  var sampleVectors=sample.map(a=>a._2).map(b=>b.param).mkString("\n")
  val sampleSentiment=sample.map(a=>a._2).map(b=>bestModel.scoreSentence(b).forward()).map(c=>(10*c).toInt).mkString("\n")

  val regex="DenseVector".r
  val regex2="\\(".r
  val regex3 ="\\)".r
  val regex4=" ".r

  sampleVectors=regex.replaceAllIn(sampleVectors,"")
  sampleVectors=regex2.replaceAllIn(sampleVectors,"")
  sampleVectors=regex3.replaceAllIn(sampleVectors,"")
  sampleVectors=regex4.replaceAllIn(sampleVectors,"")



  //scala.tools.nsc.io.File("wordLabels.csv").writeAll(wordLabels)
  scala.tools.nsc.io.File("sampleVectors.csv").writeAll(sampleVectors)
  scala.tools.nsc.io.File("stats.csv").writeAll(statsString)
  scala.tools.nsc.io.File("sampleSentiment.csv").writeAll(sampleSentiment)

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