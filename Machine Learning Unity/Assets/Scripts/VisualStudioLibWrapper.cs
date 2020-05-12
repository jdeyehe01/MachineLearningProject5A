using System.Runtime.InteropServices;

public static class VisualStudioLibWrapper
{
    [DllImport("Machine_Learning_Lib")]
    public static extern int my_add(int x, int y);

    [DllImport("Machine_Learning_Lib")]
    public static extern int my_mul(int x, int y);
}