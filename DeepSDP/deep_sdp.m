function [bound, time,status] =  deep_sdp(net,x_min,x_max,c,repeated,options)

% Author: Mahyar Fazlyab
% email: fazlyabmahyar@gmail.com, mahyarfa@seas.upenn.edu
% Website: http://www.seas.upenn.edu/~mahyarfa
% May 2020; Last revision: 12-May-2020

version = '1.0';

%%------------- BEGIN CODE --------------


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
[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(x_min,x_max);


% num_layers = length(biases);

% input dimension
dim_in = dims(1);

% output dimension
dim_out = dims(end);

dim_last_hidden = dims(end-1);

% total number of neurons
num_neurons = sum(dims(2:end-1));


% Ip = intersect(find(Y_min>0),find(Y_max>=0));
% In = intersect(find(Y_min<=0),find(Y_max<=0));
% Ipn = setdiff(1:num_neurons,union(Ip,In));

Ip = find(Y_min>1e-3);
In = find(Y_max<-1e-3);
Ipn = setdiff(1:num_neurons,union(Ip,In));

%% Construct Min
if(strcmp(language,'cvx'))
    
    eval(['cvx_solver ' solver])
    
    if(verbose)
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end
    
    variable tau(dim_in,1) nonnegative;
elseif(strcmp(language,'yalmip'))
    tau = sdpvar(dim_in,1);
    constraints = [tau(:)>=0];
else
    
end

P = [-2*diag(tau) diag(tau)*(x_min+x_max);(x_min+x_max).'*diag(tau) -2*x_min.'*diag(tau)*x_max];
tmp = ([eye(dim_in) zeros(dim_in,num_neurons+1);zeros(1,dim_in+num_neurons) 1]);
Min = tmp.'*P*tmp;

%% QC for activation functions

if(strcmp(activation,'relu'))
    T = zeros(num_neurons);
    
    %     T11 = T;
    %     T12 = T;
    %     T22 = T;
    %     if(repeated)
    %
    %         II = eye(num_neurons);
    %         C = nchoosek(1:num_neurons,2);
    %
    %
    %         alpha_param = (X_min(C(:,1))-X_min(C(:,2)))./(Y_min(C(:,1))-Y_min(C(:,2)));
    %         beta_param = (X_max(C(:,1))-X_max(C(:,2)))./(Y_max(C(:,1))-Y_max(C(:,2)));
    %
    %         m = num_neurons*(num_neurons-1)/2;
    %
    %         if(strcmp(language,'cvx'))
    %             variable zeta(m,1) nonnegative;
    %         end
    %
    %
    %         if(strcmp(language,'yalmip'))
    %             zeta = sdpvar(m,1);
    %             constraints = [constraints, zeta(:)>=0];
    %         end
    %
    %         E = II(:,C(:,1))-II(:,C(:,2));
    %         T11 = -2*E*diag(zeta.*alpha_param.*beta_param)*E';
    %         T12 = E*diag(zeta.*(alpha_param+beta_param))*E';
    %         T22 = -2*E*diag(zeta)*E';
    %     end
    
    if(repeated)
        II = eye(num_neurons);
        C = [];
        if(numel(Ip)>1)
            C = nchoosek(Ip,2);
        end
        if(numel(In)>1)
            C = [C;nchoosek(In,2)];
        end
        C = nchoosek(1:num_neurons,2);
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
    s = (X_max-X_min)./(Y_max-Y_min);
    
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


%% Construct Mmid
A = ([blkdiag(weights{1:end-1}) zeros(num_neurons,dim_last_hidden)]);
B = ([zeros(num_neurons,dim_in) eye(num_neurons)]);
bb = cat(1,biases{1:end-1});
CM_mid = ([A bb;B zeros(size(B,1),1);zeros(1,size(B,2)) 1]);
Mmid = CM_mid.'*Q*CM_mid;

%% Construct Mout
if(strcmp(language,'cvx'))
    variable b;
elseif(strcmp(language,'yalmip'))
    b = sdpvar(1);
end

obj = b;

S = [zeros(dim_out,dim_out) c;c' -2*b];
tmp = ([zeros(dim_out,dim_in+num_neurons-dim_last_hidden) weights{end} biases{end};zeros(1,dim_in+num_neurons) 1]);
Mout = tmp.'*S*tmp;


%% solve SDP

if(strcmp(language,'cvx'))
    
    minimize(obj)
    
    subject to
    
    Min+Mmid+Mout<=0;
    
    nu(In)>=0;
    nu(Ipn)>=0;
    
    eta(Ip)>=0;
    eta(Ipn)>=0;
    
    
    cvx_end
    
    bound = obj;
    time= cvx_cputime;
    status = cvx_status;
    
%     X = rect2d(x_min,x_max);
%     figure;
%     scatter(X(1,:),X(2,:));hold on;
%     E = ellipsoid2d(2*diag(tau),-diag(tau)*(x_min+x_max),2*x_min'*diag(tau)*x_max);
%     scatter(E(1,:),E(2,:));
    
    
elseif(strcmp(language,'yalmip'))
    constraints = [constraints, Min+Mmid+Mout<=0];
    options = sdpsettings('solver',solver,'verbose',verbose);
    out = optimize(constraints,obj,options);
    
    bound = value(obj);
    time= out.solvertime;
    status = out.info;
    
end

%message = ['method: DeepSDP', '(version ', version, ')','| solver: ', solver, '| bound: ', num2str(bound), '| solvetime: ', num2str(time), '| status: ', status];
message = ['method: DeepSDP ', version,'| solver: ', solver, '| bound: ', num2str(bound,'%.3f'), '| solvetime: ', num2str(time,'%.3f'), '| status: ', status];



disp(message);

end