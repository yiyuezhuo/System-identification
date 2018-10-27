a = idinput(63,'prbs');
fid = fopen('IdentSeq.txt','w');
fprintf(fid,'%g\n',a);
fclose(fid);