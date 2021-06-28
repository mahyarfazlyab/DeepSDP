clc;
clear all;
clf;
addpath('DeepSDP_Ellipsoid');

%% initial setup
rng('default');

warning off;

%% neural net
num_hidden_units_per_layer = [10;30;50];


% cvx,yalmip
options.language = 'cvx';

% mosek,sedumi,sdpt3
options.solver = 'mosek';

% 0,1
options.verbose = 0;


for i=1:numel(num_hidden_units_per_layer)
    dims = [2;num_hidden_units_per_layer(i);2];
    %net = generate_random_net(dims,'relu');
    net = nnsequential(dims,'relu');
   
    %% input uncertainty (l_inf perturbation)
    prob = 0.95;
    rho = chi2inv(prob,net.dims(1));
    
    mu_x = randn(net.dims(1),1);
    Sigma_x = diag(rand(net.dims(1),1));
    
    % p-level input confidence ellipsoid
    E_in = ellipsoid_obj(mu_x,Sigma_x*rho);
    
    X_sample = repmat(mu_x,1,1e4) + sqrtm(Sigma_x)*randn(net.dims(1),1e4);
    Y_sample = net.eval(X_sample);
    
    Xin = E_in.grid();
    Xout = net.eval(Xin);
    
    %scatter(Xin(1,:),Xin(2,:),'DisplayName','Output');axis equal;hold on;
    
    
    repeated = 1;
    E_out = deep_sdp_ellipsoid(net,E_in,repeated,options);
    
    
    subplot(2,3,i);
    scatter(Y_sample(1,:),Y_sample(2,:));hold on;axis equal;
    E_out.plot('red');
    title(['n_1=' num2str(num_hidden_units_per_layer(i))]);
    
    
    subplot(2,3,i+3);
    scatter(Xout(1,:),Xout(2,:),'DisplayName','Output');axis equal;hold on;
    E_out.plot('red');
    title(['n_1=' num2str(num_hidden_units_per_layer(i))]);
    
    
    
end

