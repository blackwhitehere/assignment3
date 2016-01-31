package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.DenseMatrix

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
  def epochHook(model:Model, iter: Int, accLoss: Double): (Int,Double,Double,Double) = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName)))
    val rtn = (iter,accLoss,100 * Evaluator(model, trainSetName), 100*Evaluator(model, validationSetName))
    rtn
  }
  val learningRate = Seq(0.005,0.01) //grid
  val vectorRegularizationStrength = Seq(0.01,0.05) //grid
  val matrixRegularizationStrength = Seq(0.01,0.05)
  val wordDim = Seq(15,20)
  val hiddenDim = Seq(10,15)

  val trainSetName = "train"
  val validationSetName = "dev"
  val ParamPentas = collection.mutable.HashMap[(Double,Double,Double,Int,Int),Double]()

  for (learn<-learningRate;
       vecReg<-vectorRegularizationStrength;
       matReg<-matrixRegularizationStrength;
       wrdDim<-wordDim;
       hidDim<-hiddenDim){
    val model: Model = new RecurrentNeuralNetworkModel(wrdDim, hidDim, vecReg, matReg)
    //model.matrixParams.foreach(a=>a._2.initialize(()=>random.nextGaussian()*0.5))
    val stats=StochasticGradientDescentLearner(model, trainSetName, 2, learn, epochHook)
    ParamPentas += (vecReg,matReg,learn,wrdDim,hidDim) -> stats.last._4 //validation error rate on last epoch

  }
  val (bestVecReg, bestMatReg, bestLearn,bestWrdDim,bestHidDim)= ParamPentas.maxBy(_._2)._1
  println("the best vector regularization parameter is %4.2f\t, best matrix reg parameter is %4.2f\t, best learning rate is %4.2f\t best word dimension is %4d\t, best hidden word dimension is %4d\t".format(
    bestVecReg, bestMatReg, bestLearn,bestWrdDim,bestHidDim)
  )
  val bestModel: Model = new RecurrentNeuralNetworkModel(bestWrdDim, bestHidDim, bestVecReg, bestMatReg)
  val stats=StochasticGradientDescentLearner(bestModel, trainSetName, 10, bestLearn, epochHook)

  val statsStr=stats.mkString("\n")
  scala.tools.nsc.io.File("stats.csv").writeAll(statsStr)

  Predictor(bestModel, "test")

  CleanAndWrite(bestModel)

  /*for ((paramName, paramBlock) <- model.matrixParams) {
    println(s"$paramName:\n${paramBlock.param}\n")
  }*/



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