using AsistenteVirualML.Datos;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLServices
{
    public class MLService : IMLService
    {
        private MLContext _mlContext;
        private static ITransformer _modeloDeEntrenamiento;
        private PredictionEngine<ItemCompra, ItemCompraPrediccion> _motorDePrediccion;

        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "datos-prueba.tsv");

        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");



        public MLService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public IDataView ObtenerDatosDeEntrenamiento()
        {
            IDataView _datosDeEntrenamiento;


            SqlConnectionStringBuilder connectionStringBuilder = new();
            connectionStringBuilder.DataSource = ".";
            connectionStringBuilder.InitialCatalog = "machineLearning";
            connectionStringBuilder.IntegratedSecurity = true;
            var cs = connectionStringBuilder.ConnectionString;
            var dataView = new DataView();

            using (SqlConnection connection = new SqlConnection(cs))
            {
                connection.Open();
                SqlCommand cmd = connection.CreateCommand();
                cmd.CommandText = "SELECT * FROM Producto";
                var dataAdapter = new SqlDataAdapter(cmd);
                var dataSet = new DataSet();
                dataAdapter.Fill(dataSet);
                var dataTable = dataSet.Tables[0];
                dataView = new DataView(dataTable);
                connection.Close();

            }

            _datosDeEntrenamiento = _mlContext.Data.LoadFromEnumerable(dataView.Cast<DataRowView>().Select(row => new ItemCompra
            {
                UserId = row["UserId"].ToString(),
                EventId = row["EventId"].ToString(),
                Personas = row["Personas"].ToString(),
                Product = row["Product"].ToString()
            }));

            return _datosDeEntrenamiento;
        }




        public IEstimator<ITransformer> ProcesarDatos()
        {
            Console.WriteLine($"=============== Procesando Datos ===============");

            var tuberia = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Product", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "EventId", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Personas", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);
            
            Console.WriteLine($"=============== Finalizando Procesamiento de Datos ===============");

            return tuberia;

        }

        public void Entrenar(IDataView datosEntrenamiento, IEstimator<ITransformer> tuberia)
        {

            // Verificar si el archivo del modelo existe
            if (File.Exists(_modelPath))
            {
                Console.WriteLine($"Loading model from: {_modelPath}");
                _modeloDeEntrenamiento = _mlContext.Model.Load(_modelPath, out var modelSchema);
                _motorDePrediccion = _mlContext.Model.CreatePredictionEngine<ItemCompra, ItemCompraPrediccion>(_modeloDeEntrenamiento);
            }
            else
            {
                // Entrenar el modelo y guardar el archivo del modelo

                var tuberiaDeEntrenamiento = tuberia.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


                Console.WriteLine($"=============== Training the model  ===============");


                _modeloDeEntrenamiento = tuberiaDeEntrenamiento.Fit(datosEntrenamiento);

                Console.WriteLine($"=============== Finalizó el entrenamiento del modelo. Hora de finalización: {DateTime.Now.ToString()} ===============");


                Console.WriteLine($"=============== Predicción única modelo recién entrenado ===============");


                _motorDePrediccion = _mlContext.Model.CreatePredictionEngine<ItemCompra, ItemCompraPrediccion>(_modeloDeEntrenamiento);


                ItemCompra item = new ItemCompra()
                {
                    EventId = "1",
                    Personas = "4"
                };

                var prediction = _motorDePrediccion.Predict(item);

                Console.WriteLine($"=============== Predicción única modelo recién entrenado - Resultado: {prediction.Product} ===============");

            }

        }

        public void Evaluar(DataViewSchema datosEntrenamiento)
        {
            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Starting time: {DateTime.Now.ToString()} ===============");


            var testDataView = _mlContext.Data.LoadFromTextFile<ItemCompra>(_testDataPath, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_modeloDeEntrenamiento.Transform(testDataView));

            Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ===============");

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");


        }

        public List<ScoreEntry> PredecirItem()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            ItemCompra singleIssue = new ItemCompra() { EventId = "1", Personas = "4" };

            _motorDePrediccion = _mlContext.Model.CreatePredictionEngine<ItemCompra, ItemCompraPrediccion>(loadedModel);

            var prediction = _motorDePrediccion.Predict(singleIssue);

            var scoreEntries = ObtenerPuntuacionesConEtiquetasOrdenadas(_motorDePrediccion.OutputSchema, "Score", prediction.Score);

            var result = new List<ScoreEntry>();
            foreach (var scoreEntry in scoreEntries)
            {
                result.Add(new ScoreEntry
                {
                    Product = scoreEntry.Key,
                    Score = scoreEntry.Value
                });
            }


            return result;
        }




        private static Dictionary<string, float> ObtenerPuntuacionesConEtiquetasOrdenadas(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = new Dictionary<string, float>();

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }

        public void GuardarModeloComoArchivo(DataViewSchema datosDeEntrenamientoEsquema)
        {

            _mlContext.Model.Save(_modeloDeEntrenamiento, datosDeEntrenamientoEsquema, _modelPath);


            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
