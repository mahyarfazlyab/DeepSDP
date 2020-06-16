function Y = fwd_prop(net,X)

activation = net.activation;
num_layers = length(net.dims)-2;
Y = X;

if(strcmp(activation,'relu'))
    for l=1:num_layers
        Y = max(0,net.weights{l}*Y+repmat(net.biases{l}(:),1,size(Y,2)));
    end
elseif(strcmp(activation,'sigmoid'))
    for l=1:num_layers
        Y = sigmf((net.weights{l})*Y+repmat(net.biases{l}(:),1,size(Y,2)),[1,0]);
    end
elseif(strcmp(activation,'tanh'))
    for l=1:num_layers
        Y = tanh((net.weights{l})*Y+repmat(net.biases{l}(:),1,size(Y,2)));
    end
end
Y = net.weights{end}*Y+repmat(net.biases{end}(:),1,size(Y,2));

end