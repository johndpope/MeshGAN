function [ ] = GetFeature( srcfolder, number )
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
cmdline = ['.\exe\ARAP.exe 3 ',srcfolder,' ', num2str(number)];
dos(cmdline);
tarfvt = [srcfolder,'\fv.mat'];
movefile('E:\SIGA2014\workspace\fv.mat',tarfvt);
%movefile('F:\SIGA2014\workspace\fv.mat',tarfvt);
end

