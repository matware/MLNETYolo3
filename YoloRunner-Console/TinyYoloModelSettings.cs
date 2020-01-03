namespace ConsoleApp2
{
    public struct TinyYoloModelSettings
    {
        // for checking Tiny yolo2 Model input and  output  parameter names,
        //you can use tools like Netron, 
        // which is installed by Visual Studio AI Tools

        // input tensor name
        public const string ModelInput = "000_net";

        // output tensor name

        public const string ModelOutput = "082_convolutional";
        //public const string ModelOutput = "094_convolutional";
        //public const string ModelOutput = "106_convolutional";
        //"082_convolutional", 13 x 13
        //"094_convolutional", 26 x 26
        //"106_convolutional", 52 x 52
    }
}
