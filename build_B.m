function B = kron(n, NQBIT)
e{1} = [1 0;0 1];            %identity
e{2} = [0 1;1 0];            %sigma x
e{3} = [0 -1i;1i 0];         %sigma y
e{4} = [1 0;0 -1];           %sigma z
A = cell(4, 4);
for j=1:4
    for m=1:4   
        A{j,m} = kron(e{j},e{m});
    end
end

B = cell(n-1, 4, 4);


switch NQBIT
case 'two qubits'
    for j=1:4
        for m=1:4
            B{1, m, j} = sparse(A{m, j});
        end
    end

case 'n>2'
    pre_slots = e{1};
    pos_slots = e{1};
    for s=1:n-3
        pre_slots = kron(e{1}, pre_slots);
    end
    for s=1:n-3
        pos_slots = kron(e{1}, pos_slots);
    end
    
    for j=1:4
        for m=1:4
        B{1, m, j} = sparse(kron(A{m, j}, pos_slots));
        B{n-1, m, j} = sparse(kron(pre_slots, A{m, j}));
        end
    end
    
    for k=2:n-2
        pre_slots = e{1};
        pos_slots = e{1};
        for s=2:k-1   % k=2 > 2:1 ; k=n-2 > 2:n-3 > n-3-1=n-4 > n-4+1=n-3
            pre_slots = kron(e{1}, pre_slots);
        end
        for s=2:n-k-1 % k=2 > 2:n-3 > n-3 ; k=n-2 > 2:n-(n-2)-1 > 2:1
            pos_slots = kron(e{1}, pos_slots);
        end
        for j=1:4
            for m=1:4
                partial = sparse(kron(pre_slots, A{m, j}));
                B{k, m, j} = sparse(kron(partial, pos_slots));
            end
        end
    end
end
end