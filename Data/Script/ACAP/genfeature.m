function genfeature(modelfolder)
    if (~exist([modelfolder,'\simp'],'file'))
        mkdir([modelfolder,'\simp']);
    end
    ACAPOpt(modelfolder);
    % cmd=['ACAPOpt.mexw64 ',modelfolder];
    % dos(cmd);

    % Save data in a .mat file
%     load([modelfolder,'\FeatureMat1.mat']);
%     save([modelfolder,'\simp\FeatureMat1.mat'],'LOGRNEW','S');
end