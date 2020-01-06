/// Based on the MLDotNet example code for the yolov2 detector.
/// https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx
/// onnx model was generated using 
/// https://github.com/Rapternmn/PyTorch-Onnx-Tensorrt

using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Microsoft.ML;

namespace YoloV3Detector
{
    class Program
    {
        static void Main(string[] args)
        {
            var testDataPath = @"../../../TestData";
            var modelPath = @"../../../Models";
            string tests = GetAbsolutePath(testDataPath);
            var modelFilePath = Path.Combine(modelPath, "yolov3-tiny.onnx");
            var imagesFolder = Path.Combine(tests, "images");
            var outputFolder = Path.Combine(tests, "images", "output");

            MLContext mlContext = new MLContext();
            try
            {
                IYoloConfiguration cfg = new YoloV3TinyConfig();
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);
                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext, cfg);
                var selectedOutput = cfg.Outputs[0].Name;
                Console.WriteLine($"Using ONNX output {selectedOutput}");

                // Use model to score data, pick the first output
                IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView, selectedOutput);
                Yolov3OutputParser parser = new Yolov3OutputParser(cfg, selectedOutput);

                var boundingBoxes =
                    probabilities
                    .Select(probability => parser.ParseOutputs(probability))
                    .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

                for (var i = 0; i < images.Count(); i++)
                {
                    string imageFileName = images.ElementAt(i).Label;
                    IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);
                    DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects, cfg);
                    LogDetectedObjects(imageFileName, detectedObjects);
                }

            }
            catch (Exception ex)

            {
                Console.WriteLine(ex.ToString());
            }
            Console.WriteLine("========= End of Process..Hit any Key ========");
            Console.ReadLine();
        }

        private static void DrawBoundingBox(string inputImageLocation,
            string outputImageLocation,
            string imageName,
            IList<YoloBoundingBox> filteredBoundingBoxes,
            IYoloConfiguration cfg)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;
            foreach (var box in filteredBoundingBoxes)
            {
                var x = (uint)Math.Max(box.Dimensions.X, 0);
                var y = (uint)Math.Max(box.Dimensions.Y, 0);
                var width = box.Dimensions.Width;
                var height = box.Dimensions.Height;
                x = (uint)originalImageWidth * x / cfg.ImageWidth;
                y = (uint)originalImageHeight * y / cfg.ImageHeight;
                width = (uint)originalImageWidth * width / cfg.ImageWidth;
                height = (uint)originalImageHeight * height / cfg.ImageHeight;
                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";
                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    thumbnailGraphic.DrawRectangle(pen, x - width / 2, y - height / 2, width, height);
                    if (!Directory.Exists(outputImageLocation))
                    {
                        Directory.CreateDirectory(outputImageLocation);
                    }

                    image.Save(Path.Combine(outputImageLocation, imageName));
                }
            }
        }

        private static void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
        {
            Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

            foreach (var box in boundingBoxes)
            {
                Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
            }

            Console.WriteLine("");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

    }
}
