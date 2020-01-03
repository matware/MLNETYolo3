using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
namespace YoloV3Detector
{
    public class Yolov3OutputParser
    {
        // Number of cells per box, which in yolov3 is 3, 
        public const int BoxesPerCell = 3;
        public const int BoxInfoFeatureCount = 5;
        private readonly int classCount;
        private readonly IYoloConfiguration configuration;

        public float cellWidth => configuration.ImageWidth/ configuration.RowCount;
        public float cellHeight => configuration.ImageHeight / configuration.ColumnCount;

        private int channelStride => configuration.RowCount * configuration.ColumnCount;
       
        class CellDimensions : DimensionsBase { }

        public Yolov3OutputParser(IYoloConfiguration configuration)
        {
            this.configuration = configuration;
            classCount = configuration.Labels.Length;
        }

        private static Color[] classColors = new Color[]
                {
                    Color.Khaki,
                    Color.Fuchsia,
                    Color.Silver,
                    Color.RoyalBlue,
                    Color.Green,
                    Color.DarkOrange,
                    Color.Purple,
                    Color.Gold,
                    Color.Red,
                    Color.Aquamarine,
                    Color.Lime,
                    Color.AliceBlue,
                    Color.Sienna,
                    Color.Orchid,
                    Color.Tan,
                    Color.LightPink,
                    Color.Yellow,
                    Color.HotPink,
                    Color.OliveDrab,
                    Color.SandyBrown,
                    Color.DarkTurquoise
                };

        private float Sigmoid(float value)
        {
            var k = Math.Exp(value);
            return (float)(k / (1.0 + k));
        }

        private float[] Sigmoid(float[] value)
        {
            var result = new float[value.Length];
            for (int i = 0; i < value.Length; i++)
            {
                var k = Math.Exp(value[i]);
                result[i] = (float)(k / (1.0 + k));
            }

            return result;
        }

        private float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private int GetOffset(int x, int y, int channel)
        {
            // YOLOv3 outputs a tensor that has a shape of 255x13x13, which 
            // WinML flattens into a 1D array.  To access a specific channel 
            // for a given (x,y) cell position, we need to calculate an offset
            // into the array
            return (channel * this.channelStride) + (y * configuration.ColumnCount) + x;
        }

        private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            var result =
            new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
            return result;
        }

        private float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            var offset = GetOffset(x, y, channel + 4);
            var rawConfidence = modelOutput[offset];
            var confidence = Sigmoid(rawConfidence);
            return confidence;
        }

        private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
        {
            var result = new CellDimensions
            {
                X = ((float)x + Sigmoid(boxDimensions.X)) * cellWidth,
                Y = ((float)y + Sigmoid(boxDimensions.Y)) * cellHeight,
                Width = (float)Math.Exp(boxDimensions.Width) * configuration.Anchors[box * 2],
                Height = (float)Math.Exp(boxDimensions.Height) * configuration.Anchors[box * 2 + 1],
            };
            return result;
        }

        public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
        {
            float[] predictedClasses = new float[classCount];
            int predictedClassOffset = channel + BoxInfoFeatureCount;
            for (int predictedClass = 0; predictedClass < classCount; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Sigmoid(predictedClasses);
        }

        private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
        {
            return predictedClasses
                .Select((predictedClass, index) => (Index: index, Value: predictedClass))
                .OrderByDescending(result => result.Value)
                .First();
        }

        private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;

            if (areaA <= 0)
                return 0;

            var areaB = boundingBoxB.Width * boundingBoxB.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }

        public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            int rowStride = BoxesPerCell * configuration.ColumnCount * (5 + classCount);
            int columnStride = BoxesPerCell * (classCount + BoxInfoFeatureCount);
                       
            var boxes = new List<YoloBoundingBox>();


            for (int row = 0; row < configuration.RowCount; row++)
            {
                for (int column = 0; column < configuration.ColumnCount; column++)
                {
                    for (int box = 0; box < BoxesPerCell; box++)
                    {
                        var channel = (box * (classCount + BoxInfoFeatureCount));
                        for (int i = 0; i < BoxInfoFeatureCount; i++)
                        {
                            var index = GetOffset(row,column,channel+i);
                        }
                        
                        float confidence = GetConfidence(yoloModelOutputs, row, column, channel);
                        if (confidence < threshold)
                            continue;

                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);
                        CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);
                        
                        var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;
                        if (topScore < threshold)
                            continue;
                        boxes.Add(new YoloBoundingBox()
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = mappedBoundingBox.X,
                                Y = mappedBoundingBox.Y,
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = configuration.Labels[topResultIndex],
                            BoxColor = classColors[topResultIndex% classColors.Length]
                        });
                    }
                }
            }
            return boxes;
        }

        public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                    .OrderByDescending(b => b.Box.Confidence)
                    .ToList();

            var results = new List<YoloBoundingBox>();

            for (int i = 0; i < boxes.Count; i++)

                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }
                    if (activeCount <= 0)
                        break;
                }
            return results;
        }
    }
}
