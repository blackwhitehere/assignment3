package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.{QuasiTensor, TensorLike, sum}
import breeze.numerics._

/**
 * Problem 1
 */

object GradientChecker extends App {
  val EPSILON = 1e-6

  /**
   * For an introduction see http://cs231n.github.io/neural-networks-3/#gradcheck
   *
   * This is a basic implementation of gradient checking.
   * It is restricted in that it assumes that the function to test evaluates to a double.
   * Moreover, another constraint is that it always tests by backpropagating a gradient of 1.0.
   */
  def apply[P](model: Block[Double], paramBlock: ParamBlock[P]) = {
    paramBlock.resetGradient()
    model.forward()
    model.backward(1.0)

    var avgError = 0.0

    val gradient = paramBlock.gradParam match {
      case m: Matrix => m.toDenseVector
      case v: Vector => v
    }

    /**
     * Calculates f_theta(x_i + eps)
     * @param index i in x_i
     * @param eps value that is added to x_i
     * @return
     */
    def wiggledForward(index: Int, eps: Double): Double = {
      var result = 0.0
      paramBlock.param match {
        case v: Vector =>
          val tmp = v(index)
          v(index) = tmp + eps
          result = model.forward()
          v(index) = tmp
        case m: Matrix =>
          val (row, col) = m.rowColumnFromLinearIndex(index)
          val tmp = m(row, col)
          m(row, col) = tmp + eps
          result = model.forward()
          m(row, col) = tmp
      }
      result
    }

    for (i <- 0 until gradient.activeSize) {
      //todo: your code goes here!
      val gradientExpected: Double = {
        (wiggledForward(i,EPSILON)-wiggledForward(i,-EPSILON))/(2*EPSILON)
      }

      avgError = avgError + math.abs(gradientExpected - gradient(i))

      assert(
        math.abs(gradientExpected - gradient(i)) < EPSILON,
        "Gradient check failed!\n" +
          s"Expected gradient for ${i}th component in input is $gradientExpected but I got ${gradient(i)}"
      )
    }

    println("Average error: " + avgError)
  }

  /**
    * A very silly block to test if gradient checking is working.
    * Will only work if the implementation of the Dot block is already correct
    */

  val a = VectorParam(12)
  val b = VectorParam(12)
  /*
  //Dot product
  val simpleBlock1 = Dot(a,b)
  GradientChecker(simpleBlock1, b)

  //Sigmoid
  val simpleBlock2 = Sigmoid(Dot(a,b))
  GradientChecker(simpleBlock2, b)

  //loglikelihood
  val simpleBlock3 = NegativeLogLikelihoodLoss(Sigmoid(Dot(a,b)),1.0)
  GradientChecker(simpleBlock3, b)

  //l2
  val mat=MatrixParam(15,15)
  val simpleBlock4 = L2Regularization(0.1,Mul(mat,b))
  GradientChecker(simpleBlock4,mat)*/

  //sum of words
  val model = new SumOfWordVectorsModel(15,0.1)
  val word1 =model.wordToVector("the")
  val word2 = model.wordToVector("my")
  val sentence = model.wordVectorsToSentenceVector(Seq(word1,word2))
  val score = model.scoreSentence(sentence)
  val l2reg = model.regularizer(Seq(word1,word2))

  /*GradientChecker(score,model.vectorParams("the"))
  GradientChecker(l2reg,model.vectorParams("the"))
  GradientChecker(model.loss(Seq("my","the"),false),model.vectorParams("the"))*/

  //rNN

  val rnn= new RecurrentNeuralNetworkModel(12,12,0.1,0.01)
  val word11 =rnn.wordToVector("the")
  val word22= rnn.wordToVector("my")
  val sentence2 = rnn.wordVectorsToSentenceVector(Seq(word11,word22))
  val score2 = rnn.scoreSentence(sentence2)
  val l2reg2 =rnn.regularizer(Seq(word11,word22))

  //GradientChecker(score2,rnn.vectorParams("the"))
  //GradientChecker(l2reg2,rnn.vectorParams("the"))
  //GradientChecker(rnn.loss(Seq("my","the"),false),rnn.vectorParams("the"))

  //Mul
  val mat = MatrixParam(15,15)
  val block =Dot(Mul(mat,b),a)
  GradientChecker(Dot(sentence2,word22),rnn.vectorParams("the"))
}
