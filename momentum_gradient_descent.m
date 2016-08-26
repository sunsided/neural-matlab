function [ gd ] = momentum_gradient_descent( L, varargin )
%MOMENTUM_GRADIENT_DESCENT Initializes an momentum-based gradient descent.

    p = inputParser;
    addRequired(p,  'L', @iscell);
    addParameter(p, 'learning_rate', 0.1);
    addParameter(p, 'momentum',      0.01);
    parse(p, L, varargin{:});
    
    eta  = p.Results.learning_rate;
    my   = p.Results.momentum;

    
    
    % prepare the algorithm
    state = struct(...
                    'previous_weight_changes', [], ...
                    'eta', eta, ...
                    'my', my ...
                    );
    
    % for the first iteration, initialize previous
    % iteration weights to zero
    state.previous_weight_changes = ...
        cellfun( @(L) zeros( size(L.theta) ), L, ...
                'UniformOutput', false);
                
    % return the actual descent function
    gd = @gradient_descent;
    return;
    
    
    
    function [L] = gradient_descent(L, weight_changes)
    %GRADIENT_DESCENT Implements the actual gradient descent.
        
        for l=1:numel(L)
            L{l}.theta = L{l}.theta ...
                - state.eta * weight_changes{l} ...
                - state.my  * state.previous_weight_changes{l};

            % Store the delta as the previous delta for momentum descent
            state.previous_weight_changes{l} = weight_changes{l};
        end
        
    end
end

