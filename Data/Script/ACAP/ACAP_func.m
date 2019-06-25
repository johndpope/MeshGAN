function [ fmlogdr, fms ] = ACAP_func(partfolder)
    LOGRNEW=dlmread([partfolder,'\LOGRNEW.txt']);
    S=dlmread([partfolder,'\S.txt']);
    v_num = size(LOGRNEW, 2)/9;
    pointnum = v_num;
    if size(LOGRNEW,2)==1
        LOGRNEW = reshape(LOGRNEW,v_num*9,[])';
    end
    if size(S,2)==1
        S = reshape(S,v_num*9,[])';
    end
    [ fmlogdr, fms ] = FeatureMap( LOGRNEW, S );
    fmlogdr=permute(reshape(fmlogdr,size(fmlogdr,1),3,pointnum),[1,3,2]);
    fms=permute(reshape(fms,size(fms,1),6,pointnum),[1,3,2]);
end