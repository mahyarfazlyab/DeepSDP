function E_out = deep_sdp_ellipsoid(net,E_in,repeated,options)

% Author: Mahyar Fazlyab
% email: fazlyabmahyar@gmail.com, mahyarfa@seas.upenn.edu
% Website: http://www.seas.upenn.edu/~mahyarfa
% Last revision: 10-August-2020

version = '1.0';

%%------------- BEGIN CODE --------------

if(~strcmp(net.activation,'relu'))
    err('Only ReLU activations are supported. Other activations are under development.');
end

% if(strcmp(options.language,'yalmip'))
%     err('Only CVX is currently supported. YALMIP is under development.');
% end

if(isempty(options.language))
    language = 'cvx';
end

language = options.language;
solver = options.solver;
verbose = options.verbose;

weights = net.weights;
biases = net.biases;
activation = net.activation;
dims = net.dims;

%%
mu = E_in.mu;
Sigma = E_in.Sigma;
Sigma_inv = Sigma^-1;

x_min = mu - sqrt(diag(Sigma));
x_max = mu + sqrt(diag(Sigma));

[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(x_min,x_max);

%num_layers = length(biases);

% input dimension
dim_in = dims(1);

% output dimension
dim_out = dims(end);

dim_last_hidden = dims(end-1);

% total number of neurons
num_neurons = sum(dims(2:end-1));

Ip = find(Y_min>0);
In = find(Y_max<0);
Ipn = setdiff(1:num_neurons,union(Ip,In));


%% x0 = E0*x xl = El*x
E0 = [eye(dim_in) zeros(dim_in,num_neurons)];
El = [zeros(dim_last_hidden,dim_in+num_neurons-dim_last_hidden) eye(dim_last_hidden)];

if(strcmp(language,'cvx'))
    
    eval(['cvx_solver ' solver])
    
    if(verbose)
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end
    
    variable tau nonnegative;
elseif(strcmp(language,'yalmip'))
    tau = sdpvar(1,1);
    constraints = [tau>=0];
else
    err('only cvx and yalmip are supported.')
end

%% Construct Min
P = tau*[-Sigma_inv Sigma_inv*mu;mu'*Sigma_inv -mu'*Sigma_inv*mu+1];
CM_in = ([E0 zeros(dim_in,1);zeros(1,dim_in+num_neurons) 1]);
Min = CM_in.'*P*CM_in;

%% QC for activation functions

if(strcmp(activation,'relu'))
    T = zeros(num_neurons);
    if(repeated)
        II = eye(num_neurons);
        C = [];
        if(numel(Ip)>1)
            C = nchoosek(Ip,2);
        end
        if(numel(In)>1)
            C = [C;nchoosek(In,2)];
        end
        % C = nchoosek(1:num_neurons,2);
        m = size(C,1);
        if(m>0)
            if(strcmp(language,'cvx'))
                variable zeta(m,1) nonnegative;
            elseif(strcmp(language,'yalmip'))
                zeta = sdpvar(m,1);
                constraints = [constraints,zeta>=0];
            else
            end
            E = II(:,C(:,1))-II(:,C(:,2));
            T = E*diag(zeta)*E';
        end
    end
    
    if(strcmp(language,'cvx'))
        
        % x=max(0,y)
        
        
        % multiplier corresponding to x>=0
        variable nu(num_neurons,1) %nonnegative
        
        % sector bounds on relu x^2=xy
        variable lambda(num_neurons,1)
        
        % multipliers correspoding to x>=y
        variable eta(num_neurons,1) %nonnegative
        
        % multipliers corresponding to $lb<=x<=ub$
        variable D(num_neurons,num_neurons) diagonal nonnegative
        
    elseif(strcmp(language,'yalmip'))
        
        nu = sdpvar(num_neurons,1);
        
        lambda = sdpvar(num_neurons,1);
        
        eta = sdpvar(num_neurons,1);
        
        
        D = diag(sdpvar(num_neurons,1));
        
        constraints = [constraints, nu(In)>=0, nu(Ipn)>=0, eta(Ip)>=0, eta(Ipn)>=0, D(:)>=0];
        
    else
        error('please select "yalmip" or "cvx" for the field "language"');
    end
    
    
    %
    
    alpha_param = zeros(num_neurons,1);
    alpha_param(Ip)=1;
    
    beta_param = ones(num_neurons,1);
    beta_param(In) = 0;
    
    Q11 = -2*diag(alpha_param.*beta_param)*(diag(lambda));
    Q12 = diag(alpha_param+beta_param)*(diag(lambda))+T;
    Q13 = -nu;
    Q22 = -2*diag(lambda)-2*D-2*T;
    Q23 = nu+eta+D*(X_min+X_max);
    Q33 = -2*X_min'*D*X_max;
    
    Q = [Q11 Q12 Q13; Q12.' Q22 Q23;Q13.' Q23.' Q33];
else
    error('The method deep sdp is currently supported for ReLU activation functions only.');
end

%% net in compact form Bx = phi(Ax+b)
A_net = ([blkdiag(weights{1:end-1}) zeros(num_neurons,dim_last_hidden)]);
B_net = ([zeros(num_neurons,dim_in) eye(num_neurons)]);
b_net = cat(1,biases{1:end-1});

%% Construct Mmid
CM_mid = ([A_net b_net;B_net zeros(size(B_net,1),1);zeros(1,size(B_net,2)) 1]);
Mmid = CM_mid.'*Q*CM_mid;

if(strcmp(language,'cvx'))
    variable A(dim_out,dim_out) symmetric
    variable b(dim_out,1)
elseif(strcmp(language,'yalmip'))
    A = sdpvar(dim_out,dim_out,'symmetric');
    b = sdpvar(dim_out,1);
else
    err('only cvx and yalmip are supported.')
end

F = [A*weights{end}*El A*biases{end}+b].';
e = [zeros(dim_in+num_neurons,1);1];

% Schur Complement
M = [Min+Mmid-e*e.' F;F.' -eye(size(F,2))];

if(strcmp(language,'cvx'))
    maximize log_det(A)
    subject to
    
    M<=0;
    nu(In)>=0;
    nu(Ipn)>=0;
    eta(Ip)>=0;
    eta(Ipn)>=0;
    
    cvx_end
    
    time= cvx_cputime;
    status = cvx_status;
    
    mu_out = -A^-1*b;
    %Sigma_out = A^-2*tau^2;
    Sigma_out = A^-2;
    E_out = ellipsoid_obj(mu_out,Sigma_out);
    
    
elseif(strcmp(language,'yalmip'))
    obj = -logdet(A);
    constraints = [constraints,M<=0];
    yalmip_options = sdpsettings('solver',solver,'verbose',verbose);
    out = optimize(constraints,obj,yalmip_options);
    
    mu_out = -value(A)^-1*value(b);
    Sigma_out = value(A)^-2*value(tau)^2;
    
    time= out.solvertime;
    status = out.info;
    
    E_out = ellipsoid_obj(mu_out,Sigma_out);
    
else
    err('only cvx and yalmip are supported.');
end

message = ['method: DeepSDP_Ellipsoid ', version,'| solver: ', solver, '| solvetime: ', num2str(time,'%.3f'), '| status: ', status];

disp(message);


end