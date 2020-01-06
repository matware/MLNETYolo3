using System.Collections.Generic;

namespace YoloV3Detector
{
    public class Yolov3Config : Yolov3BaseConfig
    {
        public override string ModelInput => "000_net";
        public override IList<YoloOutput> Outputs => new List<YoloOutput>() {
            new YoloOutput(){Name="106_convolutional", Dimension=52,Anchors= new float[] {10, 13, 16, 30, 33, 23 }},
            new YoloOutput(){Name="094_convolutional", Dimension=26,Anchors= new float[] {30, 61, 62, 45, 59, 119 }},
            new YoloOutput(){Name="082_convolutional", Dimension=13,Anchors= new float[] {116, 90, 156, 198, 373, 326 }}
        };
    }

    public class YoloOutput
    {
        public string Name { get; set; }
        public uint Dimension { get; set; }
        public float[] Anchors { get; set; }
    }

    public class YoloV3TinyConfig : Yolov3BaseConfig
    {
        public override string ModelInput => "000_net";

        public override IList<YoloOutput> Outputs => new List<YoloOutput>() {
            new YoloOutput(){Name="016_convolutional", Dimension=13,Anchors= new float[] { 81, 82, 135, 169,  344, 319 }},
            new YoloOutput(){Name="023_convolutional", Dimension=26,Anchors= new float[] { 10, 14, 23, 27, 37, 58 }},
            };
    };
}
