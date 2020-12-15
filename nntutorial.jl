module nntutorial

using Random, LinearAlgebra, DataFrames, Plots, StatsPlots, LaTeXStrings;

export factivate, factivateprime, simpleloopednncal, matrixnncal, graddescDF, graddescsim
export setup_init_weights, init_tri_values

"""
``f(x) = \\frac{1}{1+e^{-x}}``
"""
factivate(x) = 1.0/(1.0 + exp(-x));

"""
``f'(x)`` for ``f(x) = \\frac{1}{1+e^{-x}}``
"""
factivateprime(x) = factivate(x)*(1-factivate(x))

function simpleloopednncal(n_layers,x,w,b)
    h = zeros(length(w));
    for l in 1:n_layers-1  
        if l == 1
            node_in = x;
        else 
            node_in = h;
        end
        h = zeros(size(w[l])[1]);
        for i in 1:size(w[l])[1]
            f_sum = 0.0;
            for j in 1:size(w[l])[2]
                 f_sum = f_sum + w[l][i,j]*node_in[j];
            end
            f_sum = f_sum + b[l][i];
            h[i] = factivate(f_sum); 
        end
    end
    return h
end

function matrixnncal(n_layers,x,w,b)
    h = zeros(length(w));
    for l in 1:n_layers-1  
        if l == 1
            node_in = x;
        else 
            node_in = h;
        end
        z = w[l]*node_in .+ b[l];
        h = factivate.(z);
    end
    return h
end

function graddescDF(f,f_cost,x,α,j)
    xtrace = zeros(j);
    xtrace[1] = x;
    ftrace = zeros(j);
    ftrace[1] = f(x);
    for i in 1:j-1
        xtrace[i+1] = xtrace[i] - α*f_cost(xtrace[i]);
        ftrace[i+1] = f(xtrace[i+1])
    end
    DF = DataFrame(x=xtrace,f_x=ftrace)
    return DF
end

function graddescsim(x_vals,f,f_cost,x,α,j,anim_name;fps_val=10)
    y_vals = [f(x) for x in x_vals];
    DF = graddescDF(f,f_cost,x,α,j);
    p = plot(x_vals,y_vals,legend=false,title=L"\alpha = %$α, x_0 = %$x");
    anim = @animate for i ∈ 1:nrow(DF)
        plot!(p,DF[1:i,1],DF[1:i,2],m=:dot);
    end
    gif(anim, anim_name, fps = fps_val);
end

function setup_init_weights(nn_structure)
    W = Dict();
    b = Dict();
    for l = 2:length(nn_structure)
        push!(W,l=>randn(nn_structure[l],nn_structure[l-1]))
        push!(b,l=>randn(nn_structure[l]))
    end
    return W, b
end

function init_tri_values(nn_structure)
    tri_W = Dict();
    tri_b = Dict();
    for l = 2:length(nn_structure)
        push!(tri_W,l=>zeros(nn_structure[l],nn_structure[l-1]))
        push!(tri_b,l=>zeros(nn_structure[l]))
    end
    return tri_W, tri_b
end

end