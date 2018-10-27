function [val,i,j,t,k] = min4d(mat)
    [V, I] = min(mat);
    I = squeeze(I);
    [V2, I2] = min(squeeze(V));
    I2 = squeeze(I2);
    [V3, I3] = min(squeeze(V2));
    I3 = squeeze(I3);
    [V4, I4] = min(squeeze(V3));
    k = I4;
    t = I3(k);
    j = I2(t,k);
    i = I(j,t,k);
    val = V4;
end