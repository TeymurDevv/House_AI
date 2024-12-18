using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Diagnostics;

public class Program
{
    public static void Main(string[] args)
    {
        // Step 1: Initialize ML.NET context
        var mlContext = new MLContext();

        // Step 2: Load the dataset
        string dataPath = Path.Combine(Environment.CurrentDirectory, "updated_housing_400.csv");
        Console.WriteLine($"Loading data from: {dataPath}");

        if (!File.Exists(dataPath))
        {
            Console.WriteLine($"Error: File not found at {dataPath}");
            return;
        }

        IDataView dataView = mlContext.Data.LoadFromTextFile<HousingData>(
            path: dataPath,
            hasHeader: true,
            separatorChar: ',');

        // Step 3: Define the data processing pipeline
        var pipeline = mlContext.Transforms.NormalizeMinMax("Size")
            .Append(mlContext.Transforms.NormalizeMinMax("Rooms"))
            .Append(mlContext.Transforms.Concatenate("Features", new[] { "Size", "Rooms" }))
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price"));

        // Step 4: Timer-Controlled Training
        Console.WriteLine("Training the model for up to 2 minutes...");
        var stopwatch = Stopwatch.StartNew();

        ITransformer model = null;
        try
        {
            while (stopwatch.Elapsed < TimeSpan.FromSeconds(120))
            {
                model = pipeline.Fit(dataView);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Training stopped due to an error: {ex.Message}");
            stopwatch.Stop();
            return;
        }

        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F2} seconds.");

        if (model == null)
        {
            Console.WriteLine("Model training failed.");
            return;
        }

        // Step 5: Test a single prediction
        var predictionEngine = mlContext.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

        var newHouse = new HousingData { Size = 1200f, Rooms = 3f };
        var prediction = predictionEngine.Predict(newHouse);

        Console.WriteLine($"Predicted price for new house (Size: {newHouse.Size}, Rooms: {newHouse.Rooms}): {prediction.Price:F2}");

        // Step 6: Save the model
        string modelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");
        mlContext.Model.Save(model, dataView.Schema, modelPath);

        Console.WriteLine($"Model saved to: {modelPath}");
    }
}

// Define the data schema for housing data
public class HousingData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1)]
    public float Rooms { get; set; }

    [LoadColumn(2)]
    public float Price { get; set; }
}

// Define the prediction output
public class HousingPrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}