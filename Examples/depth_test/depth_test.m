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

xc_in = ones(dim_in,1);
eps = 0.1;
x_min = xc_in - eps;
x_max = xc_in + eps;

Xin = rect2d(x_min,x_max);

options.language = 'cvx';
options.solver = 'mosek';
options.verbose = false;


layer_list = 1:4;


for i=1:numel(layer_list)
    
    num_layers = layer_list(i);
    
    load(['net-' num2str(num_layers) 'L.mat']);
    
    subplot(1,4,i);
    
    disp(i);
    
    Xout = net.eval(Xin);
    data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;
    
    
    method = 'deepsdp';
    color = 'red';
    repeated = 0;
    [X_SDP,Y_SDP] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(1) = draw_2d_polytope(X_SDP,Y_SDP,color,'DeepSDP');hold on;
    
    
    method = 'sdr';
    color = 'black--';
    [X_SDR,Y_SDR] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(2) = draw_2d_polytope(X_SDR,Y_SDR,color,'SDR');hold on;
    
    legend(h);
    
    
    title(['$\ell=$' num2str(num_layers)],'Interpreter','latex');
end