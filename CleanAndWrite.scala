package uk.ac.ucl.cs.mr.statnlpbook.assignment3

import scala.util.Random

/**
 * Created by stan on 30-Jan-16.
 */
object CleanAndWrite {
  def apply(bestModel:Model):Unit = {
    val random = new Random()
    val sample = random.shuffle(bestModel.vectorParams).take(5000)
    var sampleVectors = sample.map(a => a._2).map(b => b.param).mkString("\n")
    val sampleSentiment = sample.map(a => a._2).map(b => bestModel.scoreSentence(b).forward()).map(c => (10 * c).toInt).mkString("\n")

    val regex = "DenseVector".r
    val regex2 = "\\(".r
    val regex3 = "\\)".r
    val regex4 = " ".r

    sampleVectors = regex.replaceAllIn(sampleVectors, "")
    sampleVectors = regex2.replaceAllIn(sampleVectors, "")
    sampleVectors = regex3.replaceAllIn(sampleVectors, "")
    sampleVectors = regex4.replaceAllIn(sampleVectors, "")

    //scala.tools.nsc.io.File("wordLabels.csv").writeAll(wordLabels)
    scala.tools.nsc.io.File("sampleVectors.csv").writeAll(sampleVectors)
    //scala.tools.nsc.io.File("stats.csv").writeAll(statsString)
    scala.tools.nsc.io.File("sampleSentiment.csv").writeAll(sampleSentiment)
  }
}
