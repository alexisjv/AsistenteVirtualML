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

namespace AsistenteVirualML.Servicios
{
    public class MLServicio : IMLServicio
    {
        private MLContext _mlContext;
        private static ITransformer _modeloDeEntrenamiento;
        private PredictionEngine<ItemCompra, ItemCompraPrediccion> _motorDePrediccion;
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "datos-prueba.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        public MLServicio(MLContext mlContext)
        {
            _mlContext = mlContext;
        }


        public List<ProductoPuntuacion> ObtenerProductosRecomendados()
        {
            var productosRecomendados = new List<ProductoPuntuacion>();
            var estadoEntrenamiento = ConsultarEstadoEntrenamiento();

            if (!estadoEntrenamiento)
            {
                var datosDeEntrenamiento = ObtenerDatosDeEntrenamiento();
                var tuberia = ProcesarDatos();
                Entrenar(datosDeEntrenamiento, tuberia);
                GuardarModeloComoArchivo(datosDeEntrenamiento.Schema);
                Evaluar(datosDeEntrenamiento.Schema);
            }

            productosRecomendados = PredecirItem();

            return productosRecomendados;
        }
       
        private bool ConsultarEstadoEntrenamiento()
        {
            if (File.Exists(_modelPath))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        private IDataView ObtenerDatosDeEntrenamiento()
        {
            IDataView datosEntrenamiento;
            var cadenaConexion = new SqlConnectionStringBuilder()
            {
                DataSource = ".",
                InitialCatalog = "machineLearning",
                IntegratedSecurity = true
            }.ConnectionString;

            using (var conexion = new SqlConnection(cadenaConexion))
            {
                conexion.Open();
                var consulta = "SELECT * FROM Producto";
                var adaptadorDatos = new SqlDataAdapter(consulta, conexion);
                var conjuntoDatos = new DataSet();
                adaptadorDatos.Fill(conjuntoDatos, "Producto");
                var tabla = conjuntoDatos.Tables[0];
                datosEntrenamiento = _mlContext.Data.LoadFromEnumerable(tabla.AsEnumerable().Select(fila => new ItemCompra
                {
                    UserId = fila.Field<string>("UserId"),
                    EventId = fila.Field<string>("EventId"),
                    Personas = fila.Field<string>("Personas"),
                    Product = fila.Field<string>("Product")
                }));
            }

            return datosEntrenamiento;
        }

        private IEstimator<ITransformer> ProcesarDatos()
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

        private void Entrenar(IDataView datosEntrenamiento, IEstimator<ITransformer> tuberia)
        {
            // Entrenar el modelo y guardar el archivo del modelo
            var tuberiaDeEntrenamiento = tuberia
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine($"=============== Entrenando el modelo ===============");

            _modeloDeEntrenamiento = tuberiaDeEntrenamiento.Fit(datosEntrenamiento);

            // Imprimir mensajes en la consola
            Console.WriteLine($"=============== Entrenamiento finalizado: {DateTime.Now.ToString()} ===============");

        }

        private void GuardarModeloComoArchivo(DataViewSchema datosDeEntrenamientoEsquema)
        {

            _mlContext.Model.Save(_modeloDeEntrenamiento, datosDeEntrenamientoEsquema, _modelPath);


            Console.WriteLine("The model is saved to {0}", _modelPath);
        }

        private void Evaluar(DataViewSchema datosEntrenamiento)
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

        private List<ProductoPuntuacion> PredecirItem()
        {
            // Cargar el modelo previamente entrenado
            ITransformer modeloCargado = _mlContext.Model.Load(_modelPath, out var modeloEntrada);

            // Crear un objeto de tipo ItemCompra con datos de ejemplo para hacer la predicción
            ItemCompra item = new ItemCompra()
            {
                EventId = "1",
                Personas = "4"
            };

            // Crear el motor de predicción
            _motorDePrediccion = _mlContext.Model.CreatePredictionEngine<ItemCompra, ItemCompraPrediccion>(modeloCargado);

            // Hacer la predicción
            var prediccion = _motorDePrediccion.Predict(item);

            // Obtener las puntuaciones con etiquetas ordenadas
            var puntuacionesConEtiquetas = ObtenerPuntuacionesConEtiquetasOrdenadas(_motorDePrediccion.OutputSchema, "Score", prediccion.Score);

            // Convertir los resultados a una lista de ProductoPuntuacion
            var resultado = new List<ProductoPuntuacion>();
            foreach (var puntuacionConEtiqueta in puntuacionesConEtiquetas)
            {
                resultado.Add(new ProductoPuntuacion
                {
                    Producto = puntuacionConEtiqueta.Key,
                    Puntuacion = puntuacionConEtiqueta.Value
                });
            }

            return resultado;
        }

        private static Dictionary<string, float> ObtenerPuntuacionesConEtiquetasOrdenadas(DataViewSchema schema, string nombreColumna, float[] puntuaciones)
        {
            Dictionary<string, float> resultado = new Dictionary<string, float>();

            var columna = schema.GetColumnOrNull(nombreColumna);

            var nombresDeRanuras = new VBuffer<ReadOnlyMemory<char>>();
            columna.Value.GetSlotNames(ref nombresDeRanuras);
            var nombres = new string[nombresDeRanuras.Length];
            var num = 0;
            foreach (var valorDenso in nombresDeRanuras.DenseValues())
            {
                resultado.Add(valorDenso.ToString(), puntuaciones[num++]);
            }

            return resultado.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }



    }
}
