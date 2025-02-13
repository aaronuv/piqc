function S = build_S(n)
e{1} = [0 1;1 0];            %sigma x
e{2} = [0 -1i;1i 0];         %sigma y
e{3} = [1 0;0 -1];           %sigma z
e{4}   = [1 0;0 1];          %identity

S = cell(4, 4, n, n);

for a=1:4
    for b=1:4
        for j=1:n
            for i=1:j-1
                prod=e{a};
                for s=1:i-1
                    prod=kron(e{4}, prod);
                end
                
                for s=1:j-i-1
                    prod=kron(prod, e{4});
                end
                
                prod=kron(prod,e{b});
                
                for s=1:n-j
                    prod=kron(prod, e{4});
                end
                
                S{a, b, i, j}=sparse(prod);
            end
        end
    end
end