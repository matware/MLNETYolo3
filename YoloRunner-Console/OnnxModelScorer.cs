using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace YoloV3Detector
{
    public class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;
        private readonly IYoloConfiguration configuration;
        private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext, IYoloConfiguration cfg)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
            this.configuration = cfg;
        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({configuration.ImageWidth},{configuration.ImageHeight})");
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            var outputs = (from o in configuration.Outputs select o.Name).ToArray();
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: configuration.ModelInput, imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages(outputColumnName: configuration.ModelInput
                        , imageWidth: (int)configuration.ImageWidth
                        , imageHeight: (int)configuration.ImageHeight
                        , resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill
                        ))
            .Append(mlContext.Transforms.ExtractPixels
                    (outputColumnName: configuration.ModelInput
                    , inputColumnName: configuration.ModelInput
                    , scaleImage: 1f / 255f
                    , interleavePixelColors: false))

            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation,
                outputColumnNames: outputs,
                inputColumnNames: new[] { configuration.ModelInput }, fallbackToCpu: true)
                );

            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model, string selectedOutput)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");
            IDataView scoredData = model.Transform(testData);
            IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(selectedOutput);

            return probabilities;
        }

        public IEnumerable<float[]> Score(IDataView data, string selectedOutput)
        {
            var model = LoadModel(modelLocation);

            return PredictDataUsingModel(data, model, selectedOutput);
        }
    }
}
