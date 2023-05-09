using AsistenteVirualML.Datos;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AsistenteVirualML.Servicios
{
    public interface IMLServicio
    {
        List<ProductoPuntuacion> ObtenerProductosRecomendados();
        
    }
}
