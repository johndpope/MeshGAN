function adj = neighbour2adj( neighbour )
% Transfer neighbour matrix to adjacency matrix
p_num = size(neighbour, 1);
degree = size(neighbour, 2);
adj = zeros(p_num, p_num);

for i = 1 : p_num
    for j = 1 : degree
        if neighbour(i, j) ~= 0
            adj(i, neighbour(i, j)) = 1;
            adj(neighbour(i, j), i) = 1;
        end
    end
end

end

