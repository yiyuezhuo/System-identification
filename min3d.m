function [val,i,j,t] = min3d(mat3d)
    [V, I] = min(mat3d);
    I = squeeze(I);
    [V2, I2] = min(squeeze(V));
    I2 = squeeze(I2);
    [V3, I3] = min(squeeze(V2));
    %I3 = squeeze(I3);
    t = I3;
    j = I2(t);
    i = I(j,t);
    val = V3;
end