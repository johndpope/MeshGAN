function [ logdr, s ] = recover_acap( feature )

    model_num = size(feature, 1);
    pnum = size(feature, 2);
    
    logdr = zeros(model_num, pnum * 9);
    s = zeros(model_num, pnum * 9);
    
    for i = 1:model_num
       for j = 1:pnum
           tlogdr = feature(i, j, 1:3);
           ts = feature(i, j, 4:9);
           
           offset = 9 * (j - 1);
           
           logdr(i, offset + 1) = 1;
           logdr(i, offset + 2) = tlogdr(1);
           logdr(i, offset + 3) = tlogdr(2);
           logdr(i, offset + 4) = -tlogdr(1);
           logdr(i, offset + 5) = 1;
           logdr(i, offset + 6) = tlogdr(3);
           logdr(i, offset + 7) = -tlogdr(2);
           logdr(i, offset + 8) = -tlogdr(3);
           logdr(i, offset + 9) = 1;
           
           s(i, offset + 1) = ts(1);
           s(i, offset + 2) = ts(2);
           s(i, offset + 3) = ts(3);
           s(i, offset + 4) = ts(2);
           s(i, offset + 5) = ts(4);
           s(i, offset + 6) = ts(5);
           s(i, offset + 7) = ts(3);
           s(i, offset + 8) = ts(5);
           s(i, offset + 9) = ts(6);
       end
    end

end

