package com.datarobot.drum

// h2o dependencies
import hex.genmodel.GenModel
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.MojoModel;
import hex.ModelCategory.{Regression, Binomial, Multinomial}

import scala.util.{Try, Success, Failure}
import collection.JavaConverters._

import java.util.ServiceLoader
import java.util.ArrayList
import java.net.URLClassLoader;
import java.lang.Thread
import java.io.{Reader, File, BufferedReader, FileReader, StringWriter, InputStreamReader, ByteArrayInputStream}
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
  var headers: Array[String] = null

  override def predict(inputBytes: Array[Byte]): String = {
    val predictions = Try(this.scoreReader(new BufferedReader(new InputStreamReader(new ByteArrayInputStream(inputBytes)))))
    this.predictionsToString(predictions)
  }

  override def predictUnstructured[T](
    inputBytes: Array[Byte], mimetype: String, charset: String, query: java.util.Map[String, String]
  ): T = ???

  def predictionsToString[T](predictions: Try[Array[Array[T]]]): String = {
    val csvPrinter: CSVPrinter = new CSVPrinter(
      new StringWriter(),
      CSVFormat.DEFAULT.withHeader(headers: _*)
    )

    predictions match {
      case Success(preds) =>
        preds.foreach(p =>
          csvPrinter.printRecord(p.map { _.asInstanceOf[Object] }: _*)
        )
      case Failure(e) =>
        throw new Exception(
          s"During the course of making a prediction an exception was encountered: ${e}"
        )
    }

    csvPrinter.flush()
    val outStream: StringWriter =
      csvPrinter.getOut().asInstanceOf[StringWriter];
    outStream.toString()
  }

  def scoreReader(in: Reader) = {
    val csvFormat = CSVFormat.DEFAULT.withHeader();
    val parser = csvFormat.parse(in)
    val sParser = parser.iterator.asScala.map { _.toMap }.map { map2RowData }

    val predictions = sParser.map { record =>
      val prediction = this.model.getModelCategory match {
        case Regression => Array(this.model.predictRegression(record).value)
        case Binomial   => this.model.predictBinomial(record).classProbabilities
        case Multinomial =>
          this.model.predictMultinomial(record).classProbabilities
        case _ =>
          throw new Exception(
            s"${this.model.getModelCategory} is currently not supported"
          )
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

    val pathOfJar = classOf[
      hex.genmodel.GenModel
    ].getProtectionDomain.getCodeSource.getLocation.toURI.getPath
    val compilePojo = s"javac -cp ${pathOfJar} ${pojo.getAbsolutePath}"
    val returnCode = compilePojo.!
    val urls = returnCode match {
      case 0 => Array(pojo.getAbsoluteFile.getParentFile.toURI.toURL)
      case _ =>
        throw new Exception(
          s"executing '${compilePojo}' exited with non-zero code.  " +
            s"Make sure POJO exists and hasn't been renamed.  If the POJO is > 1GB in size, consider using MOJO"
        )
    }

    val pojoName = pojo.getName.replace(".java", "")
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
    // POJO artifact must have all lower case `.java` extension, as that's what `javac` only supports.
    val re = new scala.util.matching.Regex("(.java$)|(.zip$)|(.ZIP$)")

    val filesInDir = Paths.get(modelDir).toFile().listFiles()
    val files = filesInDir.filter { f => re.findAllMatchIn(f.getName).hasNext }

    val file = files.length match {
      case 0 =>
        throw new Exception(
          s"No model artifact found in the model directory. Found files: ${filesInDir.map { _.getName }.mkString(",")}\n" +
          s"Supported model extensions: java/zip/ZIP"
        )
      case 1 => files.apply(0)
      case _ =>
        throw new Exception(
          s"More than 1 main model artifact found in model directory: ${files.map { _.getName }.mkString(",")}"
        )
    }

    val fileExt = re.findAllIn(file.getName.toLowerCase).next

    val modelConfig = fileExt match {
      case ".zip"  => modelConfigViaMojo(file)
      case ".java" => modelConfigViaPojo(file)
      case _       => throw new Exception("no usable H2O model artifact available")
    }

    val model = new EasyPredictModelWrapper(modelConfig)

    model
  }

  override def configure(
      params: java.util.Map[String, AnyRef] = new java.util.HashMap[String, AnyRef]()
  ) = {

    model = loadModel(
      params.get("__custom_model_path__").asInstanceOf[String]
    )
    customModelPath = params.get("__custom_model_path__").asInstanceOf[String]
    negativeClassLabel = params.get("negativeClassLabel").asInstanceOf[String]
    positiveClassLabel = params.get("positiveClassLabel").asInstanceOf[String]

    isRegression = model.m.getModelCategory match {
      case Regression  => true
      case Binomial    => false
      case Multinomial => false
      case _ =>
        throw new Exception(
          s"${this.model.getModelCategory} is currently not supported"
        )
    }
    headers = this.model.getModelCategory match {
    case Regression  => Array("Predictions")
    case Binomial    => this.model.getResponseDomainValues
    case Multinomial => this.model.getResponseDomainValues
    case _ =>
      throw new Exception(
        s"${this.model.getModelCategory} is currently not supported"
      )
    }
  }

}
