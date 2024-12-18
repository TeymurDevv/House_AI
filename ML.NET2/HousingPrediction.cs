using Microsoft.ML.Data;

namespace ML.NET2;

public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}