using AsistenteVirualML.Datos;
using AsistenteVirualML.Servicios;
using Microsoft.AspNetCore.Mvc;

namespace AsistenteVirtualML.Controladores
{
    [ApiController]
    [Route("(api/)")]
    public class MLControlador : Controller
    {

        private IMLServicio mLServicio;

        public MLControlador(IMLServicio mLServicio)
        {
            this.mLServicio = mLServicio;
        }

        [HttpGet]
        public IActionResult Index()
        {
            var productosRecomendados = mLServicio.ObtenerProductosRecomendados();

            return Ok(new { productosRecomendados });
        }
    }
}
