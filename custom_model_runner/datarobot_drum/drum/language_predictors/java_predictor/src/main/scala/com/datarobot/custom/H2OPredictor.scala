package com.datarobot.custom

// h2o dependencies
import hex.genmodel.GenModel
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.MojoModel;

import scala.util.{Try, Success, Failure}
import collection.JavaConverters._

import java.util.ServiceLoader
import java.util.ArrayList
import java.net.URLClassLoader;
import java.lang.Thread
import java.io.File
import java.io.{BufferedReader, StringReader, StringWriter}
import java.nio.file.Paths

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import sys.process._

class H2OPredictor(
    name: String
) extends BasePredictor(name) {

  var model: EasyPredictModelWrapper = null
  var params: java.util.HashMap[String, Any] = null
  var isRegression: Boolean = false

  var customModelPath: String = null
  var negativeClassLabel: String = null
  var positiveClassLabel: String = null

  def predict(stringCSV: String): String = {

    val csvPrinter: CSVPrinter = this.isRegression match {
      case true =>
        new CSVPrinter(
          new StringWriter(),
          CSVFormat.DEFAULT.withHeader("Predictions")
        )
      case false =>
        new CSVPrinter(
          new StringWriter(),
          CSVFormat.DEFAULT
            .withHeader(this.positiveClassLabel, this.negativeClassLabel)
        )
    }

    val predictions = Try(this.scoreStringCSV(stringCSV))

    predictions match {
      case Success(preds) => {
        isRegression match {
          case true =>
            preds.foreach(p =>
              csvPrinter.printRecord(p(0).asInstanceOf[Object])
            )
          case false =>
            preds.foreach { p =>
              csvPrinter.printRecord(
                p(1).asInstanceOf[Object],
                p(0).asInstanceOf[Object]
              )
            }
        }
      }
      case Failure(e) => throw new Exception(e)
    }

    csvPrinter.flush()
    val outStream: StringWriter =
      csvPrinter.getOut().asInstanceOf[StringWriter];
    outStream.toString()

  }

  def scoreStringCSV(stringCSV: String) = {

    val csvFormat = CSVFormat.DEFAULT.withHeader();

    val parser =
      csvFormat.parse(new BufferedReader(new StringReader(stringCSV)))

    val sParser = parser.iterator.asScala.map { _.toMap }.map { map2RowData }

    val predictions = sParser.map { record =>
      val prediction = this.isRegression match {
        case true  => Array(this.model.predictRegression(record).value)
        case false => this.model.predictBinomial(record).classProbabilities
      }
      prediction
    }.toArray
    predictions
  }

  def map2RowData(x: java.util.Map[String, String]): RowData = {
    val row = x.asScala
    val rowData: RowData = new RowData();
    row.foreach { case (k, v) => rowData.put(k, v.toString) }
    rowData
  }

  def modelConfigViaMojo(mojo: java.io.File): EasyPredictModelWrapper.Config = {
    new EasyPredictModelWrapper.Config()
      .setModel(MojoModel.load(mojo.getAbsolutePath))
      .setConvertUnknownCategoricalLevelsToNa(true)
      .setConvertInvalidNumbersToNa(true)
  }

  def modelConfigViaPojo(
      pojo: java.io.File
  ): EasyPredictModelWrapper.Config = {
    val urls = pojo.getName.endsWith("java") match {
      case true => {
        val pathOfJar = classOf[hex.genmodel.GenModel].getProtectionDomain.getCodeSource.getLocation.toURI.getPath
        s"javac -cp ${pathOfJar} ${pojo.getAbsolutePath}".!
        Array(pojo.getAbsoluteFile.getParentFile.toURI.toURL)
      }
      case false => throw new Exception("While drum was looking for a Pojo none was found.")
    }

    val pojoName = pojo.getName.replace(".jar", "").replace(".java", "")
    val urlClassLoader = URLClassLoader.newInstance(
      urls,
      Thread.currentThread().getContextClassLoader()
    )
    val h2oRawPredictor =
      urlClassLoader.loadClass(pojoName).newInstance.asInstanceOf[GenModel]
    val modelConfig = new EasyPredictModelWrapper.Config()
      .setModel(h2oRawPredictor)
      .setConvertUnknownCategoricalLevelsToNa(true)
      .setConvertInvalidNumbersToNa(true)
    modelConfig
  }

  def loadModel(modelDir: String): EasyPredictModelWrapper = {

    // h2o model artifacts
    val re = new scala.util.matching.Regex("(.java$)|(.zip$)")

    val files = Paths.get(modelDir).toFile().listFiles().filter { f =>
      re.findAllMatchIn(f.getName).hasNext
    }

    val file = files.length match {
      case 0 =>
        throw new Exception("no model artifact found in model directory")
      case 1 => files.apply(0)
      case _ =>
        throw new Exception(
          s"more than 1 main model artifact found in model directory: ${files.map { _.getName }.mkString(",")}"
        )
    }

    val fileExt = re.findAllIn(file.getName).next

    val modelConfig = fileExt match {
      case ".zip" => modelConfigViaMojo(file)
      case ".java"  => modelConfigViaPojo(file)
      case _ => throw new Exception("no usable H2O model artifact available")
    }

    val model = new EasyPredictModelWrapper(modelConfig)

    model
  }

  def configure(
      params: java.util.Map[String, Any] = new java.util.HashMap[String, Any]()
  ) = {

    model = loadModel(
      params.get("__custom_model_path__").asInstanceOf[String]
    )
    customModelPath = params.get("__custom_model_path__").asInstanceOf[String]
    negativeClassLabel = params.get("negativeClassLabel").asInstanceOf[String]
    positiveClassLabel = params.get("positiveClassLabel").asInstanceOf[String]

    isRegression = model.m.getModelCategory match {
      case hex.ModelCategory.Regression => true
      case hex.ModelCategory.Binomial   => false
      case _ =>
        throw new Exception("model is not regression or binary classification")
    }

  }

}