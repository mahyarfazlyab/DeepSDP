% Author: Mahyar Fazlyab
% email: fazlyabmahyar@gmail.com, mahyarfa@seas.upenn.edu
% Website: http://www.seas.upenn.edu/~mahyarfa
% May 2020; Last revision: 12-May-2020


%%
clc;
clear all;
clf;
addpath('../DeepSDP/');

%
rng('default');

warning off;


verbose = true;
m = 6;

dim_in = 2;
dim_out = 2;
num_hidden_units_per_layer = 500;
num_layers = 1;
dims = [dim_in num_hidden_units_per_layer*ones(1,num_layers) dim_out];



options.language = 'cvx';
options.solver = 'mosek';
options.verbose = false;

eps_list = [0.1,0.4,0.8];

net = nnsequential(dims,'relu');
save(['net-' num2str(num_layers) 'L.mat'],'net');

for i=1:numel(eps_list)
    
    disp(i);
    
    eps = eps_list(i);
    xc_in = ones(2,1);
    x_min = xc_in - eps;
    x_max = xc_in + eps;
    Xin = rect2d(x_min,x_max);
    
    subplot(1,3,i);
    
    
    Xout = net.eval(Xin);
    data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;
    
    
    method = 'deepsdp';
    color = 'red';
    repeated = 0;
    [X_SDP,Y_SDP] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(1) = draw_2d_polytope(X_SDP,Y_SDP,color,'DeepSDP');hold on;
    
    legend(h);
    
    
    title(['$\epsilon=$' num2str(eps)],'Interpreter','latex');
    
    
end