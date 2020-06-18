%% 

% Author: Mahyar Fazlyab
% email: fazlyabmahyar@gmail.com, mahyarfa@seas.upenn.edu
% Website: http://www.seas.upenn.edu/~mahyarfa
% May 2020; Last revision: 12-May-2020

% This function produces Figure 5
clc;
clear all;
clf;
addpath('../../DeepSDP/');
%%
rng('default');

warning off;

verbose = true;
m = 6;

dim_in = 2;
dim_out = 2;
num_hidden_units_per_layer = 10;

xc_in = ones(2,1);
eps = 0.1;
x_min = xc_in - eps;
x_max = xc_in + eps;
Xin = rect2d(x_min,x_max);


options.language = 'yalmip';
options.solver = 'mosek';
options.verbose = false;

layer_list = [7,8,9,10];


for i=1:numel(layer_list)
    
    num_layers = layer_list(i);
    
    
    
    %dims = [dim_in num_hidden_units_per_layer*ones(1,num_layers) dim_out];
    %net = generate_random_net(dims,'relu');
    %net = nnsequential(dims,'relu');
    %save(['net-' num2str(num_layers) 'L.mat'],'net');
    
    load(['net-' num2str(num_layers) 'L.mat']);
    
    subplot(2,2,i);
    
    disp(i);
    
    
    Xout = net.eval(Xin);
    data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;
    
    
    method = 'deepsdp';
    color = 'red';
    repeated = 0;
    [X_SDP,Y_SDP] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(1) = draw_2d_polytope(X_SDP,Y_SDP,color,'DeepSDP');hold on;
    
    
    %% change repeated to 1
    method = 'deepsdp';
    color = 'black';
    repeated = 1;
    [X_SDP_R,Y_SDP_R] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(2) = draw_2d_polytope(X_SDP_R,Y_SDP_R,color,'DeepSDP-Repeated');hold on;
    
    legend(h);
    
    pause(0.1);
    
    
    title(['$\ell=$' num2str(num_layers)],'Interpreter','latex');

end
