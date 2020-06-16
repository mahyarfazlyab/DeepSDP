classdef nnsequential
    properties
        dims
        activation
        weights
        biases
    end
    methods
        function obj = nnsequential(dims,activation)
            obj.dims = dims;
            obj.activation = activation;
            num_layers = length(obj.dims)-2;
            dim_in = dims(1);
            for i=1:num_layers+1
                % gaussian
                obj.weights{i} = 1/sqrt(dim_in)*randn(obj.dims(i+1),obj.dims(i));
                
                % change the condition number
%                 [U,S,V] = svd(obj.weights{i});
%                 EIG = diag(S);
%                 sig_max = 3;
%                 sig_min = 1;
%                 S(S~=0) = (sig_max-sig_min)./(max(EIG)-min(EIG)).*EIG + (min(EIG)*sig_max-max(EIG)*sig_min)/(min(EIG)-max(EIG));
%                 obj.weights{i} = U*S*V';
                
                obj.biases{i} = 1/sqrt(dim_in)*randn(obj.dims(i+1),1);
                
                % uniform
                %obj.weights{i} = 2*rand(obj.dims(i+1),obj.dims(i))-1;
                %obj.biases{i} = 2*randn(obj.dims(i+1),1)-1;
            end
            
        end
        
        function Y = eval(obj,X)
            Y = X;
            num_layers = length(obj.dims)-2;
            if(strcmp(obj.activation,'relu'))
                for l=1:num_layers
                    Y = max(0,obj.weights{l}*Y+repmat(obj.biases{l}(:),1,size(Y,2)));
                end
            elseif(strcmp(obj.activation,'sigmoid'))
                for l=1:num_layers
                    %Y = sigmf((obj.weights{l})*Y+repmat(obj.biases{l}(:),1,size(Y,2)),[1,0]);
                    Y = 1./(1+exp((obj.weights{l})*Y+repmat(obj.biases{l}(:),1,size(Y,2))));
                end
            elseif(strcmp(obj.activation,'tanh'))
                for l=1:num_layers
                    Y = tanh((obj.weights{l})*Y+repmat(obj.biases{l}(:),1,size(Y,2)));
                end
            end
            Y = obj.weights{end}*Y+repmat(obj.biases{end}(:),1,size(Y,2));
        end
        
        function Y = activate(obj,X)
            
            if(strcmp(obj.activation,'relu'))
                Y = max(X,0);
            elseif(strcmp(obj.activation,'tanh'))
                Y = tanh(Y_min);
            elseif(strcmp(X.activation,'sigmoid'))
                Y = 1./(1+exp(X));
            end
        end
        
        function [Y_min,Y_max,X_min,X_max,out_min,out_max] = interval_arithmetic(obj,x_min,x_max)
            %%
            num_layers = length(obj.dims)-2;
            
            X_min{1} = x_min;
            X_max{1} = x_max;
            
            for i=1:num_layers
                Y_min{i} = max(obj.weights{i},0)*X_min{i}+min(obj.weights{i},0)*X_max{i}+obj.biases{i}(:);
                Y_max{i} = max(obj.weights{i},0)*X_max{i}+min(obj.weights{i},0)*X_min{i}+obj.biases{i}(:);
                
                X_min{i+1} = obj.activate(Y_min{i});
                X_max{i+1} = obj.activate(Y_max{i});
                
            end
            
            i = num_layers+1;
            
            out_min = max(obj.weights{i},0)*X_min{i}+min(obj.weights{i},0)*X_max{i}+obj.biases{i}(:);
            out_max = max(obj.weights{i},0)*X_max{i}+min(obj.weights{i},0)*X_min{i}+obj.biases{i}(:);
            
            X_min = cat(1,X_min{2:end});
            X_max = cat(1,X_max{2:end});
            Y_min = cat(1,Y_min{:});
            Y_max = cat(1,Y_max{:});
        end
        
        
    end
end