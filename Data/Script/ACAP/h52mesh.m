function h52mesh(h5path,basemesh, save_path)
    % ACAP edition
	feature = h5read(h5path, '/test_mesh');
    feature = permute(feature, [3, 2, 1]);
    [logdr, s] = recover_acap(feature);
    for i = 1:size(logdr, 1)
        recon(basemesh,[save_path,'\',sprintf('%05d', i), '.obj'], logdr(i,:), s(i,:));
    end
end
