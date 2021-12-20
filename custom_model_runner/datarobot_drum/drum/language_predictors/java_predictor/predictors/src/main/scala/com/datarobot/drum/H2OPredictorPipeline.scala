package com.datarobot.drum

import ai.h2o.mojos.runtime.MojoPipeline;
import ai.h2o.mojos.runtime.frame.MojoFrame;
import ai.h2o.mojos.runtime.frame.MojoFrameBuilder;
import ai.h2o.mojos.runtime.frame.MojoRowBuilder;
import ai.h2o.mojos.runtime.lic.LicenseException;
import java.nio.file.{Path, Paths}
import java.io.{Reader, File, BufferedReader, FileReader, StringWriter, InputStreamReader, ByteArrayInputStream}

import scala.util.{Try, Success, Failure}

import collection.JavaConverters._
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

class H2OPredictorPipeline(name: String) extends BasePredictor(name) {

  var mojoPipeline: MojoPipeline = null
  var params: java.util.HashMap[String, Any] = null

  var isRegression: Boolean = false
  var customModelPath: String = null
  var negativeClassLabel: String = null
  var positiveClassLabel: String = null
  var classLabels: Array[String] = null
  var targetType: String = null
  var headers: Array[String] = null 

  override def configure(
      params: java.util.Map[String, AnyRef] = new java.util.HashMap[String, AnyRef]()
  ) = {
    mojoPipeline = loadModel(
      params.get("__custom_model_path__").asInstanceOf[String]
    )
    customModelPath = params.get("__custom_model_path__").asInstanceOf[String]
    negativeClassLabel = params.get("negativeClassLabel").asInstanceOf[String]
    positiveClassLabel = params.get("positiveClassLabel").asInstanceOf[String]
    classLabels = params.get("classLabels").asInstanceOf[Array[String]];
    targetType = params.get("targetType").asInstanceOf[String]

    // output column name clean up.  
    // dai prepends the column name to the class labels

    val outputColumns = mojoPipeline.getOutputMeta.getColumnNames
    val origLabels = outputColumns.length match { 
      case 1 => Array("Predictions")
      case 2 => Array(negativeClassLabel, positiveClassLabel)
      case _ => classLabels
    }
    headers = outputColumns.length match { 
      case 1 => Array("Predictions")
      case _ => outputColumns.map{ c => origLabels.filter{ c.contains(_)}}.flatMap{ x => x}
    }
  }

  def scoreReader(in: Reader) = {
    val csvFormat = CSVFormat.DEFAULT.withHeader();
    val parser = csvFormat.parse(in)
    val sParser = parser.iterator.asScala.map { _.toMap }

    val frameBuilder = this.mojoPipeline.getInputFrameBuilder();

    sParser.map { row =>
      val rowBuilder = frameBuilder.getMojoRowBuilder();
      row.asScala.map { case (k, v) => rowBuilder.setValue(k, v) }
      frameBuilder.addRow(rowBuilder);
    }.toArray

    val iframe = frameBuilder.toMojoFrame();
    val oframe = this.mojoPipeline.transform(iframe);

    val outColumns = oframe.getColumnNames
    outColumns.zipWithIndex.map {
      case (name, index) =>
        oframe.getColumn(index).getDataAsStrings
    }.transpose
  }

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

  def loadModel(dir: String): MojoPipeline = {
    val mojoPath = Paths.get(dir, "pipeline.mojo")
    val mojoPipeline = MojoPipeline.loadFrom(mojoPath.toString)
    mojoPipeline
  }
}