using System.Collections.Generic;

namespace YoloV3Detector
{
    public interface IYoloConfiguration
    {
        uint ImageHeight { get; }
        uint ImageWidth { get; }
        string[] Labels { get; }
        string ModelInput { get; }
        IList<YoloOutput> Outputs { get; }
    }
}