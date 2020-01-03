﻿namespace YoloV3Detector
{
    public class Yolov3x13Config : Yolov3BaseConfig
    {
        public override float[] Anchors => new float[] { 116, 90,
                                                156, 198,
                                                373, 326 };        
        public override int RowCount => 13;
        public override int ColumnCount => 13;

        public override string ModelInput => "000_net";
        
        public override string ModelOutput => "082_convolutional";
    }

    public class Yolov3x26Config : Yolov3BaseConfig
    {
        public override float[] Anchors => new float[] { 30, 61, 62, 45, 59, 119 };
        public override int RowCount => 26;
        public override int ColumnCount => 26;
        public override string ModelInput => "000_net";
        public override string ModelOutput => "094_convolutional";

    }

    public class Yolov3x52Config : Yolov3BaseConfig
    {
        public override float[] Anchors => new float[] { 10, 13, 16, 30, 33, 23 };
        public override int RowCount => 52;
        public override int ColumnCount => 52;
        public override string ModelInput => "000_net";
        public override string ModelOutput => "106_convolutional";
    }

    public class YoloV3Tinyx13Config : Yolov3BaseConfig
    {
        public override float[] Anchors => new float[] { 81, 82, 135, 169,  344, 319 };
        public override int RowCount => 13;
        public override int ColumnCount => 13;

        public override string ModelInput => "000_net";

        public override string ModelOutput => "016_convolutional";
    }

    public class YoloV3Tinyx26Config : Yolov3BaseConfig
    {
        public override float[] Anchors => new float[] { 10, 14, 23, 27, 37, 58 };
        public override int RowCount => 26;
        public override int ColumnCount => 26;

        public override string ModelInput => "000_net";

        public override string ModelOutput => "023_convolutional";
    }
}
