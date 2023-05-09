using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AsistenteVirualML.Datos
{
    public class ItemCompra
    {
        [LoadColumn(0)]
        public string UserId { get; set; }
        [LoadColumn(1)]
        public string EventId { get; set; }
        [LoadColumn(2)]
        public string Personas { get; set; }
        [LoadColumn(3)]
        public string Product { get; set; }

    }

    public class ItemCompraPrediccion
    {
        [ColumnName("PredictedLabel")]
        public string Product;

        [ColumnName("Score")]
        public float[] Score { get; set; }

    }

    public class ProductoPuntuacion
    {
        public string Producto { get; set; }
        public float Puntuacion { get; set; }
    }

}
