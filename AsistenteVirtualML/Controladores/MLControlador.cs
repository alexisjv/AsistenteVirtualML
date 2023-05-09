using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using MLServices;

namespace AsistenteVirtualML.Controladores
{
    [ApiController]
    [Route("(api/)")]
    public class MLControlador : Controller
    {

        private IMLService mLService;

        public MLControlador(IMLService mLService)
        {
            this.mLService = mLService;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var datosDeEntrenamiento = mLService.ObtenerDatosDeEntrenamiento();
            var tuberia = mLService.ProcesarDatos();

            mLService.Entrenar(datosDeEntrenamiento, tuberia);
            mLService.GuardarModeloComoArchivo(datosDeEntrenamiento.Schema);
            mLService.Evaluar(datosDeEntrenamiento.Schema);


            var predictionResult = mLService.PredecirItem();

            return Json(new { productosRecomendados = predictionResult });
        }
    }
}
