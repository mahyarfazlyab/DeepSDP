classdef ellipsoid_obj
    properties
        mu 
        Sigma
        dim
        description = 'The ellipsoid is described by x = mu + Sigma^{0.5}*u where norm(u,2)<=1';
        
    end
    methods
        function obj = ellipsoid_obj(mu,Sigma)
            obj.mu = mu;
            obj.Sigma = Sigma;
            obj.dim = length(mu);
        end
        
        function h = plot(obj,color)
            if(length(obj.mu)~=2)
                err('the method plot is only for 2D ellipsoids.')
            else
                theta = linspace(0,2*pi,180);
                X = repmat(obj.mu,1,length(theta)) + sqrtm(obj.Sigma)*[cos(theta);sin(theta)];
                h = plot(X(1,:),X(2,:),color);
            end
        end
        
        function X = grid(obj)
            if(length(obj.mu)~=2)
                err('the method grid is only for 2D ellipsoids.')
            else
                theta = linspace(0,2*pi,180);
                r = linspace(0,1,50);
                X = [];
                for i=1:length(r)
                    X = [X repmat(obj.mu,1,length(theta)) + sqrtm(obj.Sigma)*r(i)*[cos(theta);sin(theta)]];
                end
            end
        end
        
        function Y = sample(obj,num_samples)
            %Y = obj.mu + sqrtm(obj.Sigma)*randn(length(obj.mu),num_samples);
            U = randn(obj.dim,num_samples);
            U = U./sqrt(sum(U.^2,1));
            R = nthroot(rand(num_samples,1),obj.dim)';
            U = U.*R;
            
            Y = obj.mu + sqrtm(obj.Sigma)*U;
        end
        
    end
end