function [val,i,j,t] = max3d(mat3d)
    [V, I] = max(mat3d);
    I = squeeze(I);
    [V2, I2] = max(squeeze(V));
    I2 = squeeze(I2);
    [V3, I3] = max(squeeze(V2));
    %I3 = squeeze(I3);
    t = I3;
    j = I2(t);
    i = I(j,t);
    val = V3;
end