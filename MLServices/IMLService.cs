using AsistenteVirualML.Datos;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLServices
{
    public interface IMLService
    {
        IEstimator<ITransformer> ProcesarDatos();
        void Entrenar(IDataView datosEntrenamiento, IEstimator<ITransformer> tuberia);
        void Evaluar(DataViewSchema datosEntrenamiento);
        IDataView ObtenerDatosDeEntrenamiento();

        void GuardarModeloComoArchivo(DataViewSchema datosDeEntrenamientoEsquema);

        List<ScoreEntry> PredecirItem();
        

    }
}
