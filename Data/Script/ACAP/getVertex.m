function vertex = getVertex( model_path )
    objlist=dir([model_path,'\*.obj']);
    [v1,~,~,~,~,~,~,~,~,~,~] = cotlp([model_path,objlist(1).name]);
    vertex = zeros(size(objlist, 1), size(v1, 1), size(v1, 2));
    for i = 1:size(objlist, 1)
        [v,~,~,~,~,~,~,~,~,~,~] = cotlp([model_path,objlist(i).name]);
        vertex(i, :, :) = v;
    end
        
end

