namespace YoloV3Detector
{
    public interface IYoloConfiguration
    {
        float[] Anchors { get; }
        int ColumnCount { get; }
        uint ImageHeight { get; }
        uint ImageWidth { get; }
        string[] Labels { get; }
        int RowCount { get; }
        string ModelOutput { get; }
        string ModelInput { get; }
    }
}