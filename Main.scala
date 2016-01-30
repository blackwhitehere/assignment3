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
  val learningRate = 0.01//Seq(0.005,0.01) //grid
  val vectorRegularizationStrength = 0.15//Seq(0.15,0.2) //grid
  val matrixRegularizationStrength = 0.01//Seq(0.01,0.05)
  val wordDim = 15//Seq(10,15)
  val hiddenDim = 15

  val trainSetName = "train"
  val validationSetName = "dev"
  
  val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
  val stats=StochasticGradientDescentLearner(model, trainSetName, 1, learningRate, GridSearchSOW.epochHook)
  //Predictor(model, "test")

  for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }



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