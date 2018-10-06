function [val,i,j] = max2d(mat2d)
    [V, I] = max(mat2d);
    [V2, I2] = max(V);
    i = I(I2);
    j = I2;
    val = V2;
end