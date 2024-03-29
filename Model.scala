package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import breeze.linalg.DenseVector
import uk.ac.ucl.cs.mr.statnlpbook.assignment3.LossSum

import scala.collection.mutable

/**
 * @author rockt
 */
trait Model {
  /**
   * Stores all vector parameters
   */
  val vectorParams = new mutable.HashMap[String, VectorParam]()
  /**
   * Stores all matrix parameters
   */
  val matrixParams = new mutable.HashMap[String, MatrixParam]()
  /**
   * Maps a word to its trainable or fixed vector representation
   * @param word the input word represented as string
   * @return a block that evaluates to a vector/embedding for that word
   */
  def wordToVector(word: String): Block[Vector]
  /**
   * Composes a sequence of word vectors to a sentence vectors
   * @param words a sequence of blocks that evaluate to word vectors
   * @return a block evaluating to a sentence vector
   */
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector]
  /**
   * Calculates the score of a sentence based on the vector representation of that sentence
   * @param sentence a block evaluating to a sentence vector
   * @return a block evaluating to the score between 0.0 and 1.0 of that sentence (1.0 positive sentiment, 0.0 negative sentiment)
   */
  def scoreSentence(sentence: Block[Vector]): Block[Double]
  /**
   * Predicts whether a sentence is of positive or negative sentiment (true: positive, false: negative)
   * @param sentence a tweet as a sequence of words
   * @param threshold the value above which we predict positive sentiment
   * @return whether the sentence is of positive sentiment
   */
  def predict(sentence: Seq[String])(implicit threshold: Double = 0.5): Boolean = {
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }
  /**
   * Defines the training loss
   * @param sentence a tweet as a sequence of words
   * @param target the gold label of the tweet (true: positive sentiement, false: negative sentiment)
   * @return a block evaluating to the negative log-likelihod plus a regularization term
   */
  def loss(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }
  /**
   * Regularizes the parameters of the model for a given input example
   * @param words a sequence of blocks evaluating to word vectors
   * @return a block representing the regularization loss on the parameters of the model
   */
  def regularizer(words: Seq[Block[Vector]]): Loss
}


/**
 * Problem 2
 * A sum of word vectors model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
 */
class SumOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double = 0.0) extends Model {
  /**
   * I removed dependency on lookuptable so that each time a new class object is created a new HashMap is created
   */
  override val vectorParams = mutable.HashMap[String, VectorParam]()
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  def wordToVector(word: String): Block[Vector] ={
    if (vectorParams.contains(word)) {
      vectorParams(word)
    } else {
      vectorParams.getOrElseUpdate(word, VectorParam(embeddingSize))
    }
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = Sum(words)


  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(sentence,vectorParams("param_w"))) //func_theta

  def regularizer(words: Seq[Block[Vector]]): Loss = {
    L2Regularization(regularizationStrength,words :+ vectorParams("param_w"): _*)
  }

}


/**
 * Problem 3
 * A recurrent neural network model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class RecurrentNeuralNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams = mutable.HashMap[String, VectorParam]()
  vectorParams += "param_w" -> VectorParam(hiddenSize)
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  override val matrixParams: mutable.HashMap[String, MatrixParam] = mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def wordToVector(word: String): Block[Vector] ={
    if (vectorParams.contains(word)) {
      vectorParams(word)
    } else {
      vectorParams.getOrElseUpdate(word, VectorParam(embeddingSize))
    }
  }

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] ={
    words.foldLeft(vectorParams("param_h0"):Block[Vector])((a,b)=>
      Tanh(
        Sum(
          Seq(
            Mul(matrixParams("param_Wh"),a),
            Mul(matrixParams("param_Wx"),b),
            vectorParams("param_b")
          )
        )
      )
    )
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"),sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"):_*),
      L2Regularization(matrixRegularizationStrength, matrixParams("param_Wx"), matrixParams("param_Wh"))
    )
}




/**
 * Problem 4
 * Modified recurrent neural network model
 * @param embeddingSize dimension of the word vectors used in this model
 * @param hiddenSize dimension of the hidden state vector used in this model
 * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
 * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
 */
class RecurrentNeuralNetworkModelP4(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] =
    LookupTable.trainableWordVectors
  vectorParams += "param_w" -> VectorParam(hiddenSize)
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  val p = 0.5

  override val matrixParams: mutable.HashMap[String, MatrixParam] =
    new mutable.HashMap[String, MatrixParam]()
  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def wordToVector(word: String): Block[Vector] =LookupTable.addTrainableWordVector(word,embeddingSize)

  val fileinput = io.Source.fromFile("./glove.twitter.27B.25d.txt", "utf-8") //
  fileinput.getLines().foreach(line=>{
    val words = line.split(" ")
    val vectors = words.slice(1,words.size).map(w=>w.toDouble*0.1)
    LookupTable.addTrainableWordVector(words(0),DenseVector(vectors))
  })

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {


    words.map(Dropout(p,_)).filter(_.prob > p).foldLeft(vectorParams("param_h0"):Block[Vector])((a,b)=>Tanh(Sum(Seq(Mul(matrixParams("param_Wh"),a),Mul(matrixParams("param_Wx"),b),vectorParams("param_b")))))

    //words.foldLeft(vectorParams("param_h0"): Block[Vector])((a, b) => Tanh(Sum(Seq(Mul(matrixParams("param_Wh"), a), Mul(matrixParams("param_Wx"), b), vectorParams("param_b")))))
  }
  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"),sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"):_*),
      L2Regularization(matrixRegularizationStrength, matrixParams("param_Wx"), matrixParams("param_Wh"))
    )
}