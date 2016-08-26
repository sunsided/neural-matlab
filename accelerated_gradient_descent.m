function [ gd ] = accelerated_gradient_descent( L, varargin )
%ACCELERATED_GRADIENT_DESCENT Initializes an momentum-based gradient descent.
% This method - if somebody knows the name, please tell me - tracks the
% directions of the weight changes of each parameter and provides
% individual learning rates for each of them. As long as the current
% direction and the error gradient are identical, the learning rate will
% increase until it hits maximum velocity; if the directions change, 
% the learning rate is reset to the minimum velocity.

    p = inputParser;
    addRequired(p,  'L', @iscell);
    addParameter(p, 'acceleration',     1.1);
    addParameter(p, 'min_velocity',    0.01);
    addParameter(p, 'max_velocity',      4);
    parse(p, L, varargin{:});
    
    
    % prepare the algorithm
    state = struct(...
                    'weight_changes', [], ...
                    'alpha',          p.Results.acceleration, ...
                    'v_min',          p.Results.min_velocity, ...
                    'v_max',          p.Results.max_velocity ...
                    );
    
    % for the first iteration, initialize previous
    % iteration weights to zero
    state.weight_changes = ...
        cellfun( @(L) zeros( size(L.theta) ), L, ...
                'UniformOutput', false);
                
    % return the actual descent function
    gd = @gradient_descent;
    return;
    
    
    
    function [L] = gradient_descent(L, weight_changes)
    %GRADIENT_DESCENT Implements the actual gradient descent.
        
        for l=1:numel(L)
            
            % determine the old and new directions of the gradients
            old_gradient  = state.weight_changes{l};
            new_gradient  = weight_changes{l};
            
            old_direction = sign(old_gradient);
            new_direction = sign(new_gradient);
            
            % we don't need this value, we're just initializing the
            % size of the direction variable
            direction     = old_gradient;
            
            % all gradients that still point in the same direction
            % are now being accelerated
            same = old_direction == new_direction;
            direction(same) = min( ...
                old_gradient(same) * state.alpha, ...
                state.v_max);
            
            % all gradients that are not pointing in the same direction
            % are stopped and will move slowly in the new direction.
            % This implies that if the new direction is zero, movement
            % is stopped altogether.
            direction(~same) = new_direction(~same) * state.v_min;
            
            % apply the changes
            L{l}.theta = L{l}.theta - direction;

            % Store the delta as the previous delta for momentum descent
            state.weight_changes{l} = direction;
        end
        
    end
end

