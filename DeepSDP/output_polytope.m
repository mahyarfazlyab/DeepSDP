function [X,Y] = output_polytope(net,x_min,x_max,method,repeated,options,m)

X = [];
Y = [];

for i=1:m
    theta = (i-1)/m*2*pi;
    C(i,:) = [cos(theta);sin(theta)];
    c = C(i,:)';
    if(strcmp(method,'deepsdp'))
        [bound, ~,~] = deep_sdp(net,x_min,x_max,c,repeated,options);
    elseif(strcmp(method,'sdr'))
        [bound, ~,~] =  nn_certify_sdr(net,x_min,x_max,c,options);
    elseif(strcmp(method,'milp'))
        [bound, ~,~] = deep_milp(net,x_min,x_max,c,'max');
    elseif(strcmp(method,'deepsdp_trial'))
        [bound, ~,~] = deep_sdp_trial(net,x_min,x_max,c,repeated,options);
    end
     B(i,:) = bound;
end


for i=1:m-1
    tmp = linsolve(C([i,i+1],:),B([i,i+1],1));
    X = [X;tmp(1)];
    Y = [Y;tmp(2)];
end

tmp = linsolve(C([1,m],:),B([1,m],1));
X = [X;tmp(1)];
Y = [Y;tmp(2)];

end