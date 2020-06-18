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
num_layers = 1;
num_hidden_units_per_layer_list = [100,500,1000];

eps = 0.1;
xc_in = ones(2,1);
x_min = xc_in - eps;
x_max = xc_in + eps;
Xin = rect2d(x_min,x_max);


options.language = 'cvx';
options.solver = 'mosek';
options.verbose = false;




for i=1:numel(num_hidden_units_per_layer_list)
    
    num_hidden_units_per_layer = num_hidden_units_per_layer_list(i);
    
    dims = [dim_in num_hidden_units_per_layer*ones(1,num_layers) dim_out];
    net = nnsequential(dims,'relu');
    save(['net-' num2str(num_hidden_units_per_layer) 'n.mat'],'net');
        
    disp(i);
    
    subplot(1,3,i);
    
    
    Xout = net.eval(Xin);
    data = scatter(Xout(1,:),Xout(2,:),'LineWidth',0.5,'Marker','.');hold on;
    
    
    method = 'deepsdp';
    color = 'red';
    repeated = 0;
    [X_SDP,Y_SDP] = output_polytope(net,x_min,x_max,method,repeated,options,m);
    h(1) = draw_2d_polytope(X_SDP,Y_SDP,color,'DeepSDP');hold on;
    
    
    legend(h);
    
    
    title(['$\epsilon=' num2str(num_hidden_units_per_layer) '$'],'Interpreter','latex');
    
    %set(gcf,'Units','inches');
    %screenposition = get(gcf,'Position');
    %set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
end
